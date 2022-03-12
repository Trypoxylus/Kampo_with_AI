# -*- coding: utf-8 -*-
import os
import cv2
import time
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import streamlit as st
from PIL import Image, ImageEnhance
from efficientnet.tfkeras import EfficientNetB7
import matplotlib.pyplot as plt

height = 128
width = 128
def tongue_diagnosis_from_webcam(img):
    #image =Image.open(img)
    #resized_image = image.resize((height, width))
    img_data = np.array(img)
    img_data = img_data / 255
    img_data = np.expand_dims(img_data, 0)
    label = ['tooth-marked', 'non tooth-marked']
    model=load_model('efn_2022_03_11.h5')
    pred = model(img_data, training=False)
    if pred > 0.5:
        x = 1
    else:
        x = 0
    ans = label[x]
    return ans

def tongue_diagnosis_from_uploaded(img):
    image =Image.open(img)
    resized_image = image.resize((height, width))
    img_data = np.array(resized_image)
    img_data = img_data / 255
    img_data = np.expand_dims(img_data, 0)
    label = ['tooth-marked', 'non tooth-marked']
    model=load_model('efn_2022_03_11.h5')
    pred = model(img_data, training=False)
    if pred > 0.5:
        x = 1
    else:
        x = 0
    ans = label[x]
    return ans

def himando(bmi):
    if bmi < 18.5:
        bmi_hyouka = 1
    elif 18.5 <= bmi < 25:
        bmi_hyouka = 2
    else:
        bmi_hyouka = 3
    
    return bmi_hyouka

def htn(upper_bp, lower_bp):
    if upper_bp<120 and lower_bp<80:
        bp_hyouka = 1
    elif 120<=upper_bp<130 and lower_bp<80:
        bp_hyouka = 2
    elif 130<=upper_bp<140 or 80<=lower_bp<90:
        bp_hyouka = 3
    elif 140<=upper_bp<160 or 90<=lower_bp<100:
        bp_hyouka = 4
    elif 160<=upper_bp<180 or 100<=lower_bp<110:
        bp_hyouka = 4
    else:
        bp_hyouka = 5
    return bp_hyouka

def kyojitu(bmi, upper_bp, lower_bp, q1, q2):
    bmi_hyouka = himando(bmi)
    bp_hyouka = htn(upper_bp, lower_bp)
    final_kyojitu = bmi_hyouka*3 + bp_hyouka*2 + 10 - (q1 + q2)
    return final_kyojitu

def kannetu(bmi, q3, q4):
    final_kannetu = himando(bmi)*2 + q4 - q3
    return final_kannetu



def main():
    st.markdown("## 東洋医学簡易診断アプリ")
    st.markdown("### 虚実・寒熱診断")
    st.write('病気の性質や症状の現われ方がどうかによって「熱」「寒」に分類します。例えば、ある患者さんの機能が異常亢進していたり、炎症症状がある場合は「熱証」、反対に、機能が異常衰退していたり、アトニー的症状があれば「寒証」とされます。「熱証」「寒証」は、体質と症状をひっくるめた、その時点でのその患者さんの病態に対する分類とされています。')
    st.write('また，病気を正気（からだの抵抗力）と邪気（病毒の破壊力）との戦いであると考え、正気が衰えていると、弱い邪気であっても負けて、病気になります。こういった状態を「虚証」といい、これに対して、正気が十分に強ければ、弱い邪気くらいでは、病気にならないが、強い邪気に会うと正気と邪気が激しく戦い病気になる。こういった状態を「実証」といいます。つまり、病気の勢いに対する抵抗力の強さによって「実証」か「虚証」に分けられます。')
    st.markdown('#### 虚実')
    weight = st.number_input(label='体重(kg)',
                        value=60,
                        )
    st.write('input: ', weight)
    tall = st.number_input(label='身長(cm)',
                        value=170,
                        )
    st.write('input: ', tall)
    bmi = weight/((tall/100)**2)
    st.write('BMI:', bmi)
    upper_BP = st.number_input(label='血圧（収縮期）[mmHg]',value=120)
    lower_BP = st.number_input(label='血圧（拡張期）[mmHg]',value=80)
    q1 = st.slider(label='疲れやすさ',
                    min_value=0,
                    max_value=5,
                    value=0,
                    )
    q2 = st.slider(label='気分が落ち込みやすいか',
                    min_value=0,
                    max_value=5,
                    value=0,
                    )
    
    st.markdown('#### 寒熱')
    q3 = st.slider(label='寒がりであるか',
                    min_value=0,
                    max_value=5,
                    value=0,
                    )
    q4 = st.slider(label='暑がりであるか',
                    min_value=0,
                    max_value=5,
                    value=0,
                    )
    deficiency_excess = kyojitu(bmi, upper_BP, lower_BP, q1, q2)
    cold_heat = kannetu(bmi, q3, q4)
    # 虚実：5-27,寒熱：-2-10
    fig, ax = plt.subplots()
    ax.set_xlabel("Excess and Deficiency")
    ax.set_ylabel("Cold and heat")
    ax.set_title("Excess and Deficiency + Cold and Heat plot")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axvspan(0, 1, color="yellow", alpha=0.3)
    ax.axhspan(-1, 0, color="green", alpha=0.3)
    ax.axhspan(0, 1, color="red", alpha=0.3)
    ax.axhspan(-1, 0, color="blue", alpha=0.3)
    data = [[(deficiency_excess-16)/11, (cold_heat-4)/6]]
    x, y = zip(*data)
    plt.scatter(x, y)
    st.pyplot(fig)


    st.write('虚実：',deficiency_excess, '寒熱：', cold_heat)

    st.markdown("### 舌診：tooth-marked or non tooth-marked?")
    st.write('中医学では、健康状態を確認する1つの手段として、舌を観察しています。舌からは血液の状態、水分代謝の状態、元気の状態など様々な情報が得られます。')
    
    f = st.file_uploader(label='Upload file:',type=['jpg', 'jpeg', 'png'])
    st.write('input: ', f)
    IMG_PATH = 'imgs'
    if f:
        st.markdown(f'{f.name} をアップロードしました.')
        #img_path = os.path.join(IMG_PATH, f.name)
        # 保存した画像を表示
        img = Image.open(f)
        st.image(img)
    if st.button('Diagnose from uploaded image'):
        x = tongue_diagnosis_from_uploaded(f)
        st.write(x)
    #device = st.text_input("input your video/camera device", "0")
    if st.button("Caputre"):
        cap = cv2.VideoCapture(0)
        image_loc = st.empty()
        textPlaceholder = st.empty()
        
        while cap.isOpened:
            ret, frame = cap.read()
            time.sleep(0.01)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_loc.image(img)


            st.write("Capture button clicked >>>>Starting a long computation...")
            frame = cv2.resize(frame, (128,128))
            ans = tongue_diagnosis_from_webcam(frame)
            #st.write(ans)
            textPlaceholder.text(ans)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

    #f = st.file_uploader(label='Upload file:', type='jpg')
    #st.write('input: ', f)

    #if f is not None:
    #    # X: 信頼できないファイルは安易に評価しないこと
    #    data = f.getvalue()
    #    text = data.decode('utf-8')
    #    st.write('contents: ', text)

    #if f is not None:
    #st.write('Starting a long computation...')
    #ans = tongue_diagnosis(img)
    #st.write(ans)
    #else:
    #    ans = 'No image is uploaded!'
    #    st.warning(ans)

        
    
    





if __name__ == '__main__':
    main()

