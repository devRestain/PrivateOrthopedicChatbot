from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
base_llm = ChatOpenAI(temperature=0.1)
base_embedding_function = OpenAIEmbeddings()
#이 위는 OpenAI 모델. 바꿔야 함
from typing import Optional
from operator import itemgetter
import random
#이 아래는 langchian method.
from langchain.prompts import load_prompt, PipelinePromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema import BaseOutputParser

#벡터저장소 세팅
external_knowledge = FAISS.load_local("~", base_embedding_function)

#프롬프트 세팅
role_setting_pt = PromptTemplate.from_template("""
You are a famous orthopedic speciallist, as well as a prominent medical college professor known for your gentle attitude and calm atmosphere.
You know the power of exact references and precise figures, so you have no intention to present your opinion in the absence of any appropriate references.
When faced with a certain concern, you have a habit of analyzing and interpreting it one after another, starting with basic knowledge, step by step.
When the interpreting process is over, you present a long chain of thoughts containing the whole process without abbreviation.

Now, your colleage is about to ask you some worh-considering orthopedic questions. You decided to give an accurate answer by building a chain of thoughts step by step, as you have done so far.
""")
recieve_question_pt = PromptTemplate.from_template("""
Now, your colleage asks you some worh-considering orthopedic questions. You decided to give an accurate answer by building a chain of thoughts step by step, as you have done so far.
Human: {question}
You:
""")
recieve_context_pt = PromptTemplate.from_template("""
You recieve a context about the question with the score between 0 to 1. That score present how much the context is similar with the question.
The context is much more similar with the question as the score is near 1, whereas the context has much little relevance with the question as the score is near 0.
Please make an answer of which length is three or four sentences long based on that context with considering the score.

context: {context}
score: {score}
""")
map_reduce_pt = PromptTemplate.from_template("""
You have given your colleage an answer of his question with previous context and its score.
Your colleage's question and your previous answer pair is as below:
question: {question}
answer: {previous_answer}

Now you have a new context and its score. Try to comprehend it and judge that you could improve your answer based on the new context.
If you wanna renew your answer based on the new context, do it with a step-by-step chain of thoughts.
If you don't wanna renew your answer, just answer with your previous answer as is.

question: {question}
answer:
""")
choice_pt = PromptTemplate.from_template("""
You are a machine that find the most consistent argument among given 5 arguments.
You have to read all the 5 arguments, and choose only one argument, which has most relevance with other 4 arguments, among 5 arguments as below.
When you choose one argument, say the number of the argument.
You have to answer 1, 2, 3, 4, or 5. Other answers are not permitted.
                                         
1 {ans1}
2 {ans2}
3 {ans3}
4 {ans4}
5 {ans5}

answer:
""")
fin_pt = PromptTemplate.from_template("""
You are a information processing expert who are well-known for your gentle mood and calm atmosphere.
You always think step by step from basic knowledge. You don't try to make something when you don't know about it. You'd rather say just you don't know about it.
Now you are about to be on your work.
Your work starts when you recieve two answers generated for the same question but different period.
You would compare them and present their differences.
Let's start!

answer generated on past: {past}
answer generated on present: {present}
you:
""")
instant_pt = PromptTemplate.from_template("""empty""")
Dynamic_Few_shot_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ("ai", "Based on {article_name}, which is pubulished on {journal} {year}, {answer}"),
])

Prompts_list = [
    ("role", role_setting_pt),
    ("que", recieve_question_pt),
    ("ct", recieve_context_pt),
    ("MR", map_reduce_pt),
    ("DFS", instant_pt),
]

DFS_pt_base = PromptTemplate.from_template("""
{role}
{ct}
{que}
""")
DFS_pt_MR = PromptTemplate.from_template("""
{role}
{ct}
{MR}
""")
Choice_shuffle_prompt_base = ChatPromptTemplate.from_template("""
{role}
{DFS}
{que}
""")

DFS_base_prompt = PipelinePromptTemplate(final_prompt= DFS_pt_base, pipeline_prompts= Prompts_list)
DFS_MR_prompt = PipelinePromptTemplate(final_prompt= DFS_pt_MR, pipeline_prompts= Prompts_list)

#함수 세팅
def _Similarity_search_with_year_filter(question: str, time_status: dict, how_many_search: int) -> list:
    def IsOnPeriod(metadata: dict) -> bool:
        #similarity search 결과가 원하는 기간 내에 있는지 filter하는 내부 함수
        publish_year = int(metadata["year"])
        min_year = time_status["from"]
        max_year = time_status["to"]
        if min_year < publish_year < max_year:
            return True
        if publish_year < min_year or publish_year > max_year:
            return False
    similarity_search_res = external_knowledge.similarity_search_with_score(question, k=how_many_search, filter=IsOnPeriod(), fetch_k=0)
    return similarity_search_res
def Similarity_search_with_year_filter(_dict: dict) -> list:
    """
    #Recieve a dictionary {question: str, time_status: 'past' OR 'present', how_many_search: int}.\n
    #Return a list which contains results generated by FAISS similarity search with score.\n
    #Each result is filtered by whether it is on period we need or not.\n
    #The number of the results is how_many_search.
    """
    return _Similarity_search_with_year_filter(question=_dict["question"], time_status=_dict["time_status"], how_many_search=_dict["how_many_search"])

def Modify_SimSearch_res(similarity_search_res: list) -> list:
    """
    Recieve a result of FAISS similarity search.\n
    Return a list which contains dictionaries {article_name:str, journal:str, year:str, context:list}\n
    The context list contains dictionaries {context:str, score:str} generated as a result of similarity search in the same article.
    """
    Article_information_list = list()
    instant_information_list = list()
    instant_context_list = list()
    for items in similarity_search_res:
        metadata_each = items[0].metadata
        each_article_information_tuple = (metadata_each["article_name"], metadata_each["journal"], metadata_each["year"])
        each_article_context_dict = {"context": items[0].page_content, "score":items[1]}
        each_article_context_list = [each_article_context_dict]
        try:
            article_index = instant_information_list.index(each_article_information_tuple)
            instant_context_list[article_index].append(each_article_context_dict)
        except:
            instant_information_list.append(each_article_information_tuple)
            instant_context_list.append(each_article_context_list)
            continue
    for i in range(len(instant_information_list)) :
        each_article_information_dic = dict()
        each_article_information_dic["article_name"] = instant_information_list[i][0]
        each_article_information_dic["journal"] = instant_information_list[i][1]
        each_article_information_dic["year"] = instant_information_list[i][2]
        each_article_information_dic["context"] = instant_context_list[i]
        Article_information_list.append(each_article_information_dic)
    return Article_information_list

def _Create_Dynamic_FewShot_sample(Article_information_list: list, question: str) -> list:
    def Dynamic_FewShot_generator(Article_information_list: list):
        #context list에 있는 context들을 llm에 전달해 하나의 답을 내는 내부 함수
        for items in Article_information_list:
            each_context_list = items["context"]
            chain = DFS_base_prompt | base_llm | StrOutputParser()
            answer = chain.invoke({"question": question, "context":each_context_list[0]["context"], "score":each_context_list[0]["score"], "previous_answer":""})
            if len(each_context_list) == 1:
                yield answer
                continue
            #context가 한 개 이상이면 각 context에 대해 map reduce 방식으로 llm의 답을 업데이트
            for i in range(1, len(each_context_list)):
                each_context = each_context_list[i]["context"]
                each_score = each_context_list[i]["score"]
                chain_MR = DFS_MR_prompt | base_llm | StrOutputParser()
                answer = chain_MR.invoke({"question": question, "context":each_context, "score":each_score, "previous_answer":answer})
            yield answer
    gen = Dynamic_FewShot_generator(Article_information_list)
    for items in Article_information_list:
        items["answer"] = next(gen)
        items["question"] = question
        del items["context"]
    return Article_information_list
def Create_Dynamic_FewShot_sample(_dict: dict) -> list:
    """
    Recieve a user's question and a list which contains dict. {article_name:str, journal:str, year:str, context:list}\n
    Return a list which contains dict. {article_name:str, jouranl:str, year:str, question:str, answer:str}\n
    This function actually made one answer generated by llm based on the contexts on the context list for each article
    """
    return _Create_Dynamic_FewShot_sample(Article_information_list=_dict["Article_information_list"], question=_dict["question"])

def Choice_shuffle_with_dynamic_few_shot(Dynamic_FewShot_sample_list: list) -> dict:
    """
    Recieve a list which contains properly modified dict.\n
    With values of each dict, make a few-shot example for the model.\n
    Shuffle an index of the list 5 times, so 5 diffrent few-shot examples are made.\n 
    This is because to make an inconsistency of the model's answer.
    """
    choice_shuffle_res_dict = dict()
    for i in range(1,6):
        shuffled_list = random.sample(Dynamic_FewShot_sample_list, len(Dynamic_FewShot_sample_list))
        Dynamic_Few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=Dynamic_Few_shot_example_prompt,
            examples=shuffled_list
        )
        Prompts_list[4] = ("DFS", Dynamic_Few_shot_prompt)
        Choice_shuffle_prompt = PipelinePromptTemplate(final_prompt=Choice_shuffle_prompt_base, pipeline_prompts=Prompts_list)
        chain = Choice_shuffle_prompt | base_llm | StrOutputParser()
        res = chain.invoke({"question": User_query, "context":"","score":"","previous_answer":""})
        choice_shuffle_res_dict["ans"+str(i)] = res
    return choice_shuffle_res_dict

def Choice_One_Answer(choice_shuffle_res_dict: dict):
    choice_chain = choice_pt | base_llm | StrOutputParser()
    choice_res = choice_chain.invoke(choice_shuffle_res_dict)
    res = choice_shuffle_res_dict["ans"+choice_res]
    return res

def Gather_References(reference_list_raw: list) -> list:
    reference_list = []
    for items in reference_list_raw:
        article_name = items["article_name"]
        journal_name = items["journal"]
        year_published = items["year"]
        instant_str = f"{article_name}, {journal_name} {year_published}"
        reference_list.append(instant_str)
    return reference_list

def Plus_Compare_Arguments(_dict):
    compare_chain = fin_pt | base_llm | StrOutputParser()
    res = compare_chain.invoke({"past": _dict["past"], "present": _dict["present"]})
    _dict["compare"] = res
    return _dict

def Show_Result(_dict: dict):
    past_references = " / ".join(_dict["past_refer"])
    present_references = " / ".join(_dict["present_refer"])
    print(f"""This is your question : {_dict["question"]} 
          
About that question,
On the period {_dict["past_time"]["from"]}~{_dict["past_time"]["to"]}, {_dict["past"]}
(References: {past_references})
          
By the way,
On the period {_dict["present_time"]["from"]}~{_dict["present_time"]["to"]}, {_dict["present"]}
(References: {present_references})

Conclusionally,
Comparing those two arguments, {_dict["compare"]}
""")

#유저 세팅
Start_or_not = "N"
while Start_or_not != "T":
    Year_setting = input("Write the year of present.\nAnswer in a form of 4 numbers, like 20nn")
    Circumstance_setting = input("Write the year of past you want to compare.\nAnswer in a form of 4 numbers, like 20nn")
    Period_setting = input("How long you want to figure it out?\nAnswer in a form of 1 number, like n")
    try :
        Present_period = {"from": int(Year_setting)-int(Period_setting)+1, "to": int(Year_setting)}
        Past_period = {"from": int(Circumstance_setting)-int(Period_setting)+1, "to": int(Circumstance_setting)}
    except ValueError:
        print("원하시는 연도를 네자리의 숫자로, 기간을 한자리의 숫자로 입력하세요.")
        continue
    User_query = input("Write your question")
    
    print(f"""
    Your question: {User_query}\n
    Object period: {Period_setting} years before {Year_setting}. {Present_period["from"]}~{Present_period["to"]}\n
    Compare period: {Period_setting} years before {Circumstance_setting}. {Past_period["from"]}~{Past_period["to"]}
    """)

    input("Okay, setting complete. Just press enter key.")
    Start_or_not = input("Is that right? Answer T/F. If your answer is not T, this would restart.") 
    if Start_or_not == "T":
        break
    print("restart")

Sampling_DynamicFewShot_Chain = {
    #FAISS RAG 수행
    "Article_information_list" : RunnableLambda(Similarity_search_with_year_filter) | RunnableLambda(Modify_SimSearch_res),
    "question" : itemgetter("question")
#Choice suffle with Dynamic Fewshot 수행
} | RunnableLambda(Create_Dynamic_FewShot_sample) 

Ensemble_with_ChoiceShuffle_Chain =  RunnableLambda(Choice_shuffle_with_dynamic_few_shot) | RunnableLambda(Choice_One_Answer)

Asking_Chain = {
    "past": itemgetter("past_DFS") | Ensemble_with_ChoiceShuffle_Chain,
    "past_refer": itemgetter("past_DFS") | RunnableLambda(Gather_References),
    "past_time": itemgetter("past_time"),
    "present": itemgetter("present_DFS") | Ensemble_with_ChoiceShuffle_Chain,
    "present_refer": itemgetter("present_DFS") | RunnableLambda(Gather_References),
    "present_time": itemgetter("present_time"),
    "question": itemgetter("question"),
    } | RunnableLambda(Plus_Compare_Arguments) | RunnableLambda(Show_Result)

Main_chain = {
    "past_DFS": {
        "question": itemgetter("question"),
        "how_many_search": itemgetter("how_many_search"), 
        "time_status": itemgetter("past")
        } | Sampling_DynamicFewShot_Chain,
    "past_time": itemgetter("past"),
    "present_DFS": {
        "question": itemgetter("question"),
        "how_many_search": itemgetter("how_many_search"),
        "time_status": itemgetter("present")
        } | Sampling_DynamicFewShot_Chain,
    "present_time": itemgetter("present"),
    "question": itemgetter("question"),
    } | Asking_Chain

Main_chain.invoke({
    "question": User_query,
    "how_many_search": 10,
    "past": Past_period,
    "present": Present_period,
})
