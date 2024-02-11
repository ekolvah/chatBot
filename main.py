from llama_index import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from lmqg import TransformersQG
import textwrap
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import torch


def get_context(documents):
    context = ''
    for document in documents:
        context = document.get_text()
    return context


def generate_questions_answers(context):
    model = TransformersQG('lmqg/t5-base-squad-qag')
    qa_list = []
    context_parts = textwrap.wrap(context, 512)

    for part in context_parts:
        try:
            qa = model.generate_qa(part)
            for q in qa:
                question_answer = {'context': part, 'question': q[0], 'answer': q[1]}
                qa_list.append(question_answer)
        except Exception as e:
            print(f"Exception occurred: {e}, while processing part: {part}")

    return qa_list

def construct_index(documents):
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()
    return index

def fine_tune_model(qa_list):
    model_name = "gpt-3.5-turbo"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print('-----questions_answers-----')
    print(qa_list)

    # Подготовить данные для обучения
    contexts = [item['context'] for item in qa_list if isinstance(item, dict) and 'context' in item]

    print('-----Context-----')
    print(contexts)

    questions = [item['question'] for item in qa_list if isinstance(item, dict) and 'question' in item]

    print('-----Questions-----')
    print(questions)

    answers = [{'text': item['answer'], 'answer_start': item['context'].index(item['answer'])} for item in qa_list if isinstance(item, dict) and 'answer' in item and 'context' in item]

    print('-----Answers-----')
    print(answers)

    # Токенизация
    train_encodings = tokenizer(contexts, questions, truncation=True, padding=True, max_length=512)

    # Включение ответов
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_pos = train_encodings.char_to_token(i, answers[i]['answer_start'])
        end_pos = train_encodings.char_to_token(i, answers[i]['answer_start'] + len(answers[i]['text']))

        # If start position is None, the answer passage starts with a space.
        # Shift position by one to the right.
        start_pos = start_pos if start_pos is not None else answers[i]['answer_start'] + 1

        # If end position is None, the answer passage ends with a space.
        # Shift position by one to the left.
        end_pos = end_pos if end_pos is not None else answers[i]['answer_start'] + len(answers[i]['text']) - 1

        start_positions.append(start_pos)
        end_positions.append(end_pos)

    print('-----start_positions-----')
    print(start_positions)
    print('-----end_positions')
    print(end_positions)

    train_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    train_dataset = Dataset.from_dict(train_encodings)
    # data_collator modification
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])
    # Готовим аргументы для обучения
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        learning_rate=1e-5,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        metric_for_best_model="f1",
        logging_dir='./logs',
        report_to=['tensorboard'],
    )

    # Обучение модели
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    trainer.train()

    # Сохранение модели после обучения
    model.save_pretrained("path_to_save_model")
    tokenizer.save_pretrained("path_to_save_model")

def collate_fn(features):
    batch = {}
    for key in features[0].keys():
        batch[key] = torch.stack([feature[key] for feature in features])
    return batch

# метод который использует huggingface модель bert-base-uncased для получения ответа на вопрос
def ask_RAG_ai(question):
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    response = query_engine.query(question)

    print('-----RAG answer-----')
    print(response)

def ask_fine_tined_ai(question, context):
    # Load the fine-tuned model
    model = AutoModelForQuestionAnswering.from_pretrained("path_to_save_model")
    tokenizer = AutoTokenizer.from_pretrained("path_to_save_model")

    # Create a question-answering pipeline
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    # Use the pipeline to answer the user's question
    answer = nlp({
        'question': question,
        'context': context
    })

    print('-----Fine tuned answer-----')
    print(answer['answer'])


if __name__ == '__main__':
    documents = SimpleDirectoryReader("context_data/data").load_data()
    #construct_index(documents)
    context = get_context(documents)
    qa_list = generate_questions_answers(context)
    fine_tune_model(qa_list)

    while True:
        user_question = input("Please enter your question (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        else:
            ask_RAG_ai(user_question)
            ask_fine_tined_ai(user_question, context)

