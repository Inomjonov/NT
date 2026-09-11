from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

@app.get("/")
def home():
    return {"message":"Hello, World!"}

@app.get("/hello/{name}")
def hello(name: str):
    return {"message":"Hello " + name}

@app.get("/age/{age}")
def next_year(age: int):
    return {"now ": age, "next_year ": age + 1}


@app.get("/greet")
def greet(name: str, loud: bool = False):
    message = "Hello " + name
    if loud:
        message = message.upper()
    return {"message ": message}


students = {
    1: {'id': 1, 'name': 'Ali', 'grade': 90},
    2: {'id': 2, 'name': 'Zara', 'grade': 85},
}

@app.get('/students')
def get_all_students():
    return list(students.values())

class Student(BaseModel):
    name: str
    grade: int


@app.post('/students')
def create_student(student: Student):
    new_id = max(students) + 1 if students else 1
    students[new_id] = {'id': new_id, 'name': student.name, 'grade':student.grade}

    return students[new_id]


@app.put('/students/{student_id}')
def update_student(student_id: int, student: Student):
    if student_id not in students:
        return {'error': 'student not found'}
    students[student_id] = {'id':student_id, 'name':student.name, 'grade': student.grade}

    return students[student_id]

@app.delete('/students/{student_id}')
def delete_student(student_id: int):
    if student_id not in students:
        return {'error':'student not found'}

    deleted = students.pop(student_id)
    return {'deleted': deleted}