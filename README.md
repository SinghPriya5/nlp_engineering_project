# 𓂀 Sarcasm Detection using NLP 𓂀  
<img align="right" width="400" height="400" src="https://github.com/SinghPriya5/nlp_engineering_project/blob/main/sarcasm_image.webp">

## Table of Contents  
- [Problem Statement](#problem-statement)  
- [Goal](#goal)  
- [Approach](#approach)  
- [Dataset](#dataset)  
- [Project Workflow](#project-workflow)  
  - [Data Preprocessing](#data-preprocessing)    
  - [Model Training](#model-training)  
  - [Model Evaluation](#model-evaluation)  
  - [Model Selection](#model-selection)  
  - [Deployment](#deployment)  
- [Tools & Technologies Used](#tools--technologies-used)  
- [Accuracy & Performance](#accuracy--performance)  
- [Bug or Feature Request](#bug-or-feature-request)  
- [Future Scope](#future-scope)  
- [Conclusion](#conclusion)  

---

## Problem Statement  
- **Objective**: Detect whether a given text contains sarcasm.  
- **Target Variable**: Binary classification (Sarcasm/Not Sarcasm).  
- **Features**: Text data, linguistic patterns, sentiment features, etc.  

---

## Goal  
1️⃣ **Identify Sarcasm**: Accurately classify text as sarcastic or not.  
2️⃣ **Improve NLP Models**: Use advanced NLP techniques for better performance.  
3️⃣ **Enhance Social Media Analysis**: Detect sarcasm in tweets, reviews, and comments.  
4️⃣ **Automate Moderation**: Help in filtering sarcastic content for sentiment analysis.  

---

## Approach  
✔ **Text Processing**: Tokenization, Lemmatization, Stopword Removal.  
✔ **Feature Engineering**: TF-IDF, Word Embeddings (Word2Vec, BERT).  
✔ **Modeling**: Experimenting with ML/DL models like LSTMs, CNNs, and Transformers.  
✔ **Evaluation**: Using Accuracy, Precision, Recall, and F1-score.  

---

## Dataset  
📌 **Dataset Source**: [Sarcasm Dataset](https://github.com/SinghPriya5/nlp_engineering_project/blob/main/data/Sarcasm.csv)  
📌 **Example Features**:  
- **headline**: The input text (news headline, tweet, comment).  
- **is_sarcastic**: Label (1 = Sarcasm, 0 = Non-Sarcasm).  
- **article_link**: (Optional) Source of the headline.  

---

## Project Workflow  

### 📌 Data Preprocessing  
🟢 **Lowercasing & Punctuation Removal**  
🟢 **Tokenization & Stopword Removal**  
🟢 **Lemmatization for Better Word Representation**  

### 📌 Model Training  
💡 **ML Models**: Logistic Regression, SVM, Random Forest  
💡 **DL Models**: LSTM, BiLSTM, CNN+LSTM, BERT  
💡 **Embeddings**: TF-IDF, Word2Vec, GloVe, BERT  

### 📌 Model Evaluation  
📈 **Accuracy**  
📈 **Precision & Recall**  
📈 **Confusion Matrix**  
📈 **ROC-AUC Curve**  

### 📌 Model Selection  
✔ **Final Model**: BiLSTM + Word2Vec gave the best accuracy (~94%).  

### 📌 Deployment  
✅ **Flask Web App**  
✅ **REST API Integration**  
✅ **Deployed on Render** – [Live App Link](https://github.com/SinghPriya5/Sarcasm_Detection/issues)  

---

## Tools & Technologies Used  
- **Programming Language**: Python  
- **Libraries**: NLTK, SpaCy, TensorFlow, Keras, Scikit-Learn, Flask  
- **Deployment**: Flask, Render  

---

## Accuracy & Performance  
📊 **Final Model Accuracy**: **94%**  
📊 **F1-Score**: **0.92**  
📊 **ROC-AUC Score**: **0.95**  
 

### Inserting Value & Predicted Result  
<p align="center">
  <img src="https://github.com/SinghPriya5/nlp_engineering_project/blob/main/front.png" alt="Input Text" width="700" height="600">
</p>
  
  #### Predicted Result
<p align="center">
    <img src="https://github.com/SinghPriya5/nlp_engineering_project/blob/main/predict%201.png" alt="Predicted Result" width="700" height="600">
</p>

## Bug or Feature Request  
If you find a bug or want to request a feature, please open an [issue](https://github.com/SinghPriya5/Sarcasm_Detection/issues) and describe the problem or improvement.  

---

## Future Scope  
📌 **Use Advanced Transformer Models (GPT, T5, etc.)**  
📌 **Analyze Context-based Sarcasm in Conversations**  
📌 **Expand Dataset for Generalization**  
📌 **Improve UI with Interactive Features**  

---

## Conclusion  
Sarcasm detection using NLP is a challenging yet impactful task in text analysis. With deep learning techniques like BiLSTM and transformer models, we achieved a **94% accuracy** in sarcasm classification. This project can be extended to **sentiment analysis, chatbot moderation, and social media monitoring** for real-world applications.  

---

# 🎉 Thank You! Happy Coding! 🎉  
