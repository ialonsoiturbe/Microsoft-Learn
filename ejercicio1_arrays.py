data=[50,50,47,97,49,3,53,42,26,74,82,62,37,15,70,27,36,35,48,52,63,64]
print(data)

import numpy as np

grades = np.array(data)
print(grades)

print(type(data), 'x 2:', data * 2)
print('---')
print(type(grades), 'x 2:', grades * 2)
#Multiplicar *2 una lista es hacer una nueva repetida dos veces
#Multiplicar *2 un array es multiplicar sus elementos por dos

#Veamos la forma del array
grades.shape

#El primer elemento
grades[0]

#La media
grades.mean()

# Define an array of study hours
study_hours = [10.0,11.5,9.0,16.0,9.25,1.0,11.5,9.0,8.5,14.5,15.5,13.75,9.0,8.0,15.5,8.0,9.0,6.0,10.0,12.0,12.5,12.0]

# Create a 2D array (an array of arrays)
student_data = np.array([study_hours, grades])

# display the array
student_data

#Ahora que tenemos ambas listas, veamos la media
avg_study = student_data[0].mean()
avg_grade = student[1].mean()
print('Average study hours: {:.2f}\nAverage grade: {:.2f}'.format(avg_study, avg_grade))

#Veamos tablas con Panda
import pandas as pd

df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie', 'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny','Jakeem','Helena','Ismat','Anila','Skye','Daniel','Aisha'],'StudyHours':student_data[0],'Grade':student_data[1]})

df_students 

#Datos para el valor index 5
df_students.loc[5]

#Las filas con los valores de 0 a 5 en index:
df_students.loc[0:5]

#Los datos para las 5 primeras filas en orden ordinal en cambio:
df_students.iloc[0:5]

#find the values for the columns in positions 1 and 2 in row 0
df_students.iloc[0,[1,2]]
df_students.loc[0, 'Grade']
df_students.loc[df_students['Name']=='Aisha']
df_students[df_students['Name']=='Aisha']
df_students.query['Name=="Aisha"']
df_students[df_students.Name == 'Aisha']

#Cargar los datos desde un fichero csv
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/grades.csv
df_students = pd.read_csv('grades.csv',delimiter=',',header='infer')
df_students.head()
#Para identificar qué valores son nulos, faltan:
df_students.isnull()

#La suma de todos los que faltan es:
df_students.isnull().sum()

#we can filter the DataFrame to include only rows where any of the columns (axis 1 of the DataFrame) are null
df_students[df_students.isnull().any(axis=1)]

#Reemplazamos valores con su media
df_students.StudyHours = df_students.StudyHours.fillna(df_students.StudyHours.mean())
df_students

#Eliminamos filas con valores nulos
df_students = df_students.dropna(axis=0, how='any')
df_students

#Vamos a explorar los datos en el dataframe:
# Get the mean study hours using to column name as an index
mean_study = df_students['StudyHours'].mean()

# Get the mean grade using the column name as a property (just to make the point!)
mean_grade = df_students.Grade.mean()

# Print the mean study hours and mean grade
print('Average weekly study hours: {:.2f}\nAverage grade: {:.2f}'.format(mean_study, mean_grade))

# Get students who studied for the mean or more hours
df_students[df_students.StudyHours > mean_study]

# What was their mean grade?
df_students[df_students.StudyHours > mean_study].Grade.mean()


#We can use that information to add a new column to the DataFrame that indicates whether or not each student passed.

passes  = pd.Series(df_students['Grade'] >= 60)
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)

df_students

#Para determinar qué estudiantes han pasado y cuáles no:
print(df_students.groupby(df_students.Pass)[['StudyHours', 'Grade']].mean())

# Create a DataFrame with the data sorted by Grade (descending)
df_students = df_students.sort_values('Grade', ascending=False)

# Show the DataFrame
df_students

