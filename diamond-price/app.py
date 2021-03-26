import pickle
import streamlit as st
import numpy as np

st.set_page_config(page_title="Diamond Price Prediction")

model = pickle.load(open('./models/linreg_model.sav', 'rb'))
coef_carat, coef_cut, coef_clarity = model.coef_
intercept = model.intercept_

'''
# Diamond Price Prediction with Linear Regression
Made with **streamlit** by [riztekur](https://github.com/riztekur)
***
'''

"## Diamond Features"
cut_list = ['Fair','Good','Very Good','Ideal','Premium']
clarity_list = ['I1','SI2','SI1','VS1','VS2','VVS2','VVS1','IF']
col1, col2, col3 = st.beta_columns(3)
with col1:
    carat = st.slider('Carat (x)', min_value=0.01, max_value=6.00, step=0.01, value=1.00)
with col2:
    cut = st.selectbox('Cut (y)', options=cut_list, index=0)
with col3:
    clarity = st.selectbox('Clarity (z)', options=clarity_list, index=0)

"## Model Formula"
st.write('''
$P(x,y,z)= {c1:.0f}x + {c2:.0f}y + {c3:.0f}z + {c4:.0f}$
'''.format(c1=coef_carat, c2=coef_cut, c3=coef_clarity, c4=intercept))

cut_ord = dict(zip(cut_list,[1,2,3,4,5]))
clarity_ord = dict(zip(clarity_list,[1,2,3,4,5,6,7,8]))

feature = np.asfarray([carat, cut_ord[cut], clarity_ord[clarity]]).reshape(1,-1)
result = model.predict(feature)

"## Predicted Price"
st.write("$\${price:.0f}$".format(price=result[0]))