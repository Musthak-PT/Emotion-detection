from flask import *
import pymysql
from flask import Flask, render_template, redirect, request, session

from flask import render_template, Flask
import numpy as np

import matplotlib.pyplot as plt
import cv2
import time
from PIL import Image 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


con = pymysql.connect(host='localhost', port=3306, user='root', passwd='', db='emotion')
cmd = con.cursor()
app = Flask(__name__)
app.debug = True
app.secret_key = 'abc'
app.config["IMAGE_UPLOADS"]="C:/Users/Basith/Desktop/emotion/static/images/"

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/login')
def login1():
    return render_template('login.home.html')    

@app.route('/login1')
def managerloginpage():
    return render_template('login.manager.html')

@app.route('/login2')
def employeeloginpage():
    return render_template('login.emp.html')

@app.route('/register')
def employeeregisteration():
    return render_template('register.emp.html')


############################################################################################################################

#LOGIN PAGE OF MANAGER
@app.route('/loginformanager', methods=['GET', 'POST'])
def loginofmanager():
    user=request.form['usernameform']
    pswd=request.form['passwordform']
    cmd.execute("select * from login where username='"+user+"' and password='"+pswd+"' and usertype='manager'")
    s=cmd.fetchone()
    if s is not None:
        session["muser"]=user
        return render_template('home.mng.html',name=user)

    else:
        return '''<script> 
    alert('Invalid username or password!!TRY_AGAIN');window.location='/login1';
    </script>'''


#EMPLOYEE REGISTRATION PAGE
@app.route('/employeeregistering', methods=['GET', 'POST'])
def employeeregistering():

    fullname=request.form['flname']
    user=request.form['usrname']
    email = request.form['eml']
    phonenumber = request.form['phno']
    pswd = request.form['passwd']
    cpswd = request.form['cpasswd']
    gender = request.form['genderradio']
    if(pswd==cpswd):
     emotion="none"
     cmd.execute("insert into login values(null,'"+user+"','"+pswd+"','employee')")
     id=con.insert_id()
     cmd.execute("insert into employee values('"+str(id)+"','"+fullname+"','"+email+"','"+phonenumber+"','"+gender+"','"+user+"','"+emotion+"')")
     con.commit()
     return '''<script>
        alert('Successfully created');window.location='/login2';
        </script>'''
    else:
     return '''<script>
        alert('password mismatch>>Try again');window.location='/register';
        </script>'''


#LOGIN PAGE OF EMPLOYEE
@app.route('/loginforemployee', methods=['GET', 'POST'])
def loginofemployee():
    user=request.form['usernameform2']
    pswd=request.form['passwordform2']
    cmd.execute("select * from login where username='"+user+"' and password='"+pswd+"' and usertype='employee'")
    s=cmd.fetchone()
    
    print(s)
    if s is not None:
        session["emplogin"]=user
        
        return render_template('home.emp.html',name=user)

    else:
        return '''<script> 
    alert('Invalid username or password!!TRY_AGAIN');window.location='/login2';
    </script>'''


#################################################################################################################################
@app.route('/mydetails')
def mydetails():
    user=session["emplogin"]
    cmd.execute("select * from employee where user_name='"+user+"' ")
    k=cmd.fetchone()
    print(k)
    name = k[1].replace(" ", "_")
    return render_template('emp.details.html',name=name,mail=k[2],no=k[3],gender=k[4])

@app.route('/emphome')
def employeehome():
    user=session["emplogin"]
    cmd.execute("select * from employee where user_name='"+user+"' ")
    k=cmd.fetchone()
    return render_template('home.emp.html',name=k[1])

#LOGOUT of employee
@app.route('/logout1')
def logout1():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return render_template('login.html')

#logout of manager
@app.route('/logout2')
def logout2():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return render_template('login.html')

@app.route('/record')
def record():
    return render_template('record.html')

@app.route('/dect1')
def dect1():
    
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights('model.h5')

    
    img_= cv2.imread(r"C:\Users\TOSHIBA\Desktop\emotion\upload\pic\image.jpg")
    
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')      
    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)          
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        
        cv2.putText(img_, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (184, 134, 11), 2, cv2.LINE_AA)
        result=(emotion_dict[maxindex])
        userr=session["emplogin"]
        
        
        
        print(userr)
        print(result)
        cmd.execute("update employee set emotion='"+result+"'where user_name='"+userr+"'")
        con.commit()
        cv2.imwrite('final_img_upload.jpg', img=img_ )
        return render_template('result.emp.html',result=result,name=userr)

@app.route('/dect2')
def dect2():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('c'): 
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                webcam.release()
                img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                print("Processing image...")
                img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                
                facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    prediction = model.predict(cropped_img)
                    maxindex = int(np.argmax(prediction))
                    
                    cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (184, 134, 11), 2, cv2.LINE_AA)
                    result=(emotion_dict[maxindex])
                    
                cv2.imshow('Videos', cv2.resize(frame,(600,600),interpolation = cv2.INTER_CUBIC))
                
                userr=session["emplogin"]
                
                print(userr)
                print(result)
                cmd.execute("update employee set emotion='"+result+"'where user_name='"+userr+"'")
                con.commit()
                cv2.imwrite('final_img_capture.jpg', img=frame )
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break    
            
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
            
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
    
    
    
    return render_template('result.emp.html',result=result,name=userr  )




@app.route('/deleteemployee')
def deleteemployee():
    return render_template('deleteempp.html')



    





@app.route('/updatebymanager')
def updatebymanager():
    cmd.execute("select * from employee")
    s = cmd.fetchall()
    return render_template('emp.edit.html',val=s)

@app.route('/deleteemp',methods=['POST','GET'])
def deleteemp():
    id=request.args.get('id')
    session['eid']=id
    cmd.execute("delete from employee where eid='" + str(id) + "'")
    cmd.execute("delete from login where lid='" + str(id) + "'")
    con.commit()
    return '''<script> 
                   alert('employee details is Deleted successfully');window.location='/managerhome';
                   </script>'''





@app.route('/updatebyemployee')
def updatebyemployee():
    return render_template('viewandeditofemployee.html')










if __name__=='__main__':
    app.run()
