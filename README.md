## 🆕 Personal Information Tagger Based on Named entity recognition
Named entity recognition (NER) helps you easily identify the key elements in a text, like names of people, places, brands, monetary values, and more.Extracting the main entities in a text helps sort unstructured data and detect important information, which is crucial if you have to deal with large datasets. This project has been deployed using AWS ECR and AWS EC2 Instance.

## 💽 Dataset 
XTREME is a benchmark for the evaluation of the cross-lingual generalization ability of pre-trained multilingual models that covers 40 typologically diverse languages and includes nine tasks.

## 📚 Approach 
1. Get data and properly create text and label (Can be done using https://explosion.ai/demos/displacy-ent.
2. Use trasnformer Roberta architecture for training the ner tagger
3. Use hugging face for Robereta Tokenizer
4. Train and Deploy model for use-cases

## 🚀 API 
<img width="790" alt="image" src="https://user-images.githubusercontent.com/57321948/200125543-da8286e2-dda6-4670-bf40-0c1584f97a60.png">

## 🧑‍💻 How to setup

create fresh conda environment 
```python
conda create -p ./env python=3.7 -y
```
activate conda environment
```python
conda activate ./env
```
Install requirements
```python
pip install -r requirements.txt
```

To run inferencing
```python
python app.py
```

To launch swagger ui
```python
http://localhost:8080/docs
```
## 🧑‍💻 Tech Used
1. Natural Language processing
2. Pytorch 
3. Transformer 
4. FastApi 

## Deployment
1. AWS ECR
2. AWS EC2

## 🏭 Industrial Use-cases 
1. Search and Recommendation system 
2. Content Classification 
3. Customer Support 
4. Research Paper Screening 
5. Automatically Summarizing Resumes 

## 👋 Conclusion 
We have shown how to train our own name entity tagger along with proper inplementaion of train and predict pipeline.
