import streamlit as st
import pickle 

filename_cv = 'cv-transform.pkl'
filename_model = 'spam-email-model.pkl'
cv = pickle.load(open(filename_cv, 'rb'))
model = pickle.load(open(filename_model, 'rb'))

st.title("Phân loại email spam")
msg = st.text_input("Nhập đoạn tin nhắn muốn phân loại: ")
if st.button("Phân loại"):
   data=[msg]
   vec = cv.transform(data).toarray()
   result = model.predict(vec)
   if result[0] == 0:
      st.success("Đây không phải là tin nhắn rác")
   else:
      st.error("Đây là tin nhắn rác!")
      st.image('spam-img.jpeg')
