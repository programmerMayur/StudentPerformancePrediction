from flask import Flask, render_template, request
app = Flask(__name__)
import pickle

#For sentiment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
set(stopwords.words('english'))
#end sentiment

file = open('pkl/model1.pkl','rb')
reg = pickle.load(file)
file.close()

file1 = open('pkl/dTree.pkl','rb')
dt = pickle.load(file1)
file1.close()

@app.route('/')
def homePage():
    return render_template('homePage.html')

@app.route('/simple',methods=['GET','POST']) #,methods=['GET','POST']
def simplePred():
    if request.method == "POST":
        allValueFromFORM = request.form
        print(allValueFromFORM)
        one = float(allValueFromFORM["one"])
        two = float(allValueFromFORM["two"])
        three = float(allValueFromFORM["three"])
        four = float(allValueFromFORM["four"])
        five = float(allValueFromFORM["five"])
        six = float(allValueFromFORM["six"])
        seven = float(allValueFromFORM["seven"])
        inputPara = [one,two,three,four,five,six,seven]
        markProbability = reg.predict([inputPara])[0][0]
        markProbability = round(markProbability,2)
        print("probability:",markProbability)
        return render_template('show.html', mark=markProbability)
    return render_template('simplePred.html')

@app.route('/actual',methods=['GET','POST'])
def actualPred():
    if request.method == 'POST':
        allValues = request.form

        #here we take values from form
        school = int(allValues['school'])
        sex = int(allValues['gender'])
        age = int(allValues['age'])
        address = int(allValues['address'])	
        famsize = int(allValues['famSize'])	
        Pstatus = int(allValues['pStatus'])	
        Medu = int(allValues['mEdu'])
        Fedu = int(allValues['fEdu'])
        Mjob = int(allValues['mJob'])
        Fjob = int(allValues['fJob'])
        reason = int(allValues['rSchool'])	
        guardian =  int(allValues['guardian'])
        traveltime =  int(allValues['travelTime'])
        studytime =  int(allValues['studyTime'])
        failures = int(allValues['fail'])
        schoolsup = int(allValues['eClasses'])
        famsup = int(allValues['fSupport'])
        paid =  int(allValues['pCourses'])
        activities =  int(allValues['eClass'])
        nursery	=  int(allValues['nursery'])
        higher =  int(allValues['hEdu'])
        internet =  int(allValues['internet'])
        romantic =  int(allValues['relationship'])
        famrel =  int(allValues['fRelationship'])
        freetime	=  int(allValues['fTime'])
        goout =  int(allValues['friends'])
        Dalc =  int(allValues['alco'])
        Walc =  int(allValues['weekAlco'])
        health =  int(allValues['health'])
        absences =  int(allValues['absences'])
        G1 = int(allValues['firstYear'])
        G2 = int(allValues['secondYear'])
        subject= int(allValues['subject'])



        inputData = [school, sex, age, address, famsize, Pstatus, Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2, subject]
        print(inputData)
        vall = ""
        rfor = dt.predict([inputData])
        print(rfor)
        if rfor == 0:
            vall = "Excellent"
            print(vall)
        elif rfor == 1:
            vall = "Failure"
            print(vall)
        elif rfor == 2:
            vall = "Good"
            print(vall)
        elif rfor == 3:
            vall = "Poor"
            print(vall)
        elif rfor == 4:
            vall = "Satisfactory"
            print(vall)
            
        #print(allValues)
        accuracy = 0.821656050955414
        accuracy = round(accuracy*100,2)
        return render_template('show2.html',val = vall, acc = accuracy)
    
    return render_template('actualPred.html')
@app.route('/sentiment')
def sentimentAnalysis():
    return render_template('senti2.html')

@app.route('/sentiment', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    text1 = request.form['text1'].lower()

    processed_doc1 = ' '.join([word for word in text1.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    print(sa)
    dd = sa.polarity_scores(text=processed_doc1)
    print(dd)
    compound = round((1 + dd['compound'])/2, 2)
    compound = compound * 100

    return render_template('senti2.html', final=compound, text1=text1)


@app.route('/aboutus')
def aboutUs():
    return render_template('aboutus.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug = True)
    
