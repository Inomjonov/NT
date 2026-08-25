"""
FastAPI — simple lesson (write one method, test it, then write the next)
========================================================================

Install once:
    pip install fastapi uvicorn

HOW TO TEACH THIS FILE
----------------------
1. Open TWO terminals side by side.
   In VS Code:  Terminal -> New Terminal (Ctrl+`), then click the "split
   terminal" icon: Cmd+backslash on Mac, Ctrl+Shift+5 on Windows.
2. In terminal 1 start the server ONE time and leave it running:

       uvicorn fastapi_lesson:app --reload

   `--reload` means: every time you save the file, the server restarts by
   itself. So you write one method, press Ctrl+S, and it is live immediately.
   This terminal now also prints a log line for every request you send.
3. In terminal 2 run the `curl` commands written under each step.

WHERE EXACTLY DO I RUN `curl`?
------------------------------
In terminal 2 — a NORMAL terminal, the same place where you type `pip` or `ls`.
  * NOT inside Python. If you see `>>>`, type exit() first.
  * NOT inside a notebook cell (there you would need a `!` in front).
  * The server in terminal 1 must be running, or you get "Connection refused".
  * WINDOWS: in PowerShell write `curl.exe` instead of `curl`, and put the
    whole command on ONE line — the backslash line-break at the end of a
    line only works on Mac/Linux.

THREE WAYS TO TEST
------------------
  * /docs:    open http://127.0.0.1:8000/docs — FastAPI writes this page for
              you. Every method you add appears there with a "Try it out"
              button: fill the form, press Execute, see the answer. It works
              for POST / PUT / DELETE and needs no terminal at all.
              EASIEST FOR CLASS — start here.
  * Browser:  only works for GET. Type the URL and you see the JSON.
  * curl:     terminal command, works for every method. Given in each step.

If you do not want to start a server at all, run:

    python fastapi_lesson.py

and every example below is executed and printed for you.
"""

from fastapi import FastAPI
from pydantic import BaseModel

# The app object. Every endpoint is attached to it.
app = FastAPI()


# =============================================================================
# STEP 1 — GET: the simplest API
# =============================================================================
# WHAT IS GET?
#   GET means "give me data". It changes nothing on the server. Your browser
#   sends a GET every time you type an address, so GET is the only method you
#   can test from the address bar.
#
# TELL THE STUDENTS:
#   * @app.get("/") connects the URL path "/" to the function below it.
#   * The function name is free — the path in the decorator is what matters.
#   * We return a Python dict; FastAPI turns it into JSON automatically.
#
# TEST IT:
#   browser:  http://127.0.0.1:8000/
#   curl:     curl http://127.0.0.1:8000/
#   expect:   {"message":"Hello, World!"}


@app.get("/")
def home():
    return {"message": "Hello, World!"}


# =============================================================================
# STEP 2 — GET with a value in the URL (path parameter)
# =============================================================================
# WHAT IS A PATH PARAMETER?
#   {name} inside the path is a placeholder. Whatever the user puts there is
#   passed to the function as an argument with the same name.
#
# TELL THE STUDENTS:
#   * The name in {curly braces} MUST match the argument name.
#   * Use a path parameter when the value identifies the thing you want:
#     /students/1, /users/ali, /products/42.
#
# TEST IT:
#   browser:  http://127.0.0.1:8000/hello/Ali
#   curl:     curl http://127.0.0.1:8000/hello/Ali
#   expect:   {"message":"Hello Ali"}
#   now try:  curl http://127.0.0.1:8000/hello/Zara


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": "Hello " + name}


# WHY THE TYPE HINT MATTERS (this is the FastAPI "wow" moment):
#   `age: int` is not a comment. FastAPI converts the URL text to an int, and
#   if it cannot, it answers with a clear error — you write zero if-statements.
#
# TEST IT:
#   curl http://127.0.0.1:8000/age/20      -> {"now":20,"next_year":21}
#   curl http://127.0.0.1:8000/age/abc     -> 422 error: "not a valid integer"
#
# ASK THE STUDENTS: change `age: int` to `age: str` and try `age + 1`. It
# breaks — that is exactly the bug FastAPI is protecting you from.


@app.get("/age/{age}")
def next_year(age: int):
    return {"now": age, "next_year": age + 1}


# =============================================================================
# STEP 3 — GET with a value after "?" (query parameter)
# =============================================================================
# WHAT IS A QUERY PARAMETER?
#   Everything after the "?" in a URL, joined by "&":
#       /greet?name=Ali&loud=true
#   An argument that is NOT written in the path automatically becomes one.
#
# TELL THE STUDENTS:
#   * Path parameter  = WHICH thing you want          (/students/1)
#   * Query parameter = HOW you want it: options,
#     filters, search text, sorting, page number      (?loud=true)
#   * `name: str` has no default -> required. Missing it gives a 422 error.
#   * `loud: bool = False` has a default -> optional.
#
# TEST IT:
#   curl "http://127.0.0.1:8000/greet?name=Ali"             -> Hello Ali
#   curl "http://127.0.0.1:8000/greet?name=Ali&loud=true"   -> HELLO ALI
#   curl "http://127.0.0.1:8000/greet"                      -> 422, name required
#   (keep the quotes around the URL: & means something else in the terminal)


@app.get("/greet")
def greet(name: str, loud: bool = False):
    message = "Hello " + name
    if loud:
        message = message.upper()
    return {"message": message}


# =============================================================================
# STEP 4 — A small "database" + GET all / GET one
# =============================================================================
# TELL THE STUDENTS:
#   A real project stores data in a database (PostgreSQL, MongoDB...). For the
#   lesson a dict is enough — the endpoint code looks the same either way.
#   Because it lives in memory, restarting the server resets the data.
#
# TEST IT:
#   curl http://127.0.0.1:8000/students      -> the whole list
#   curl http://127.0.0.1:8000/students/1    -> only Ali
#   curl http://127.0.0.1:8000/students/99   -> {"error":"student not found"}

students = {
    1: {"id": 1, "name": "Ali", "grade": 90},
    2: {"id": 2, "name": "Zara", "grade": 85},
}


@app.get("/students")
def get_all_students():
    return list(students.values())


@app.get("/students/{student_id}")
def get_one_student(student_id: int):
    if student_id not in students:
        return {"error": "student not found"}
    return students[student_id]


# =============================================================================
# STEP 5 — POST: create something
# =============================================================================
# WHAT IS POST?
#   POST means "here is new data, create it". The data does NOT go in the URL —
#   it travels in the request BODY as JSON. That is why you cannot test POST
#   from the browser address bar: use /docs or curl.
#
# HOW DO WE DESCRIBE THE BODY?
#   With a Pydantic class. `class Student(BaseModel)` says: the body must be a
#   JSON object with a text `name` and a whole-number `grade`. FastAPI checks
#   it before your function runs, and shows the shape at /docs.
#
# TELL THE STUDENTS:
#   * Path/query = small values in the URL. Body = the whole object.
#   * Never put a password in the URL — URLs are saved in browser history and
#     server logs. Bodies are not. That is one reason login uses POST.
#
# TEST IT:
#   curl -X POST http://127.0.0.1:8000/students \
#        -H "Content-Type: application/json" \
#        -d '{"name": "Bek", "grade": 70}'
#
#   then check it was really added:
#   curl http://127.0.0.1:8000/students
#
#   now send a wrong type on purpose and read the error message:
#   curl -X POST http://127.0.0.1:8000/students \
#        -H "Content-Type: application/json" \
#        -d '{"name": "Bek", "grade": "excellent"}'


class Student(BaseModel):
    name: str
    grade: int


@app.post("/students")
def create_student(student: Student):
    new_id = max(students) + 1 if students else 1
    students[new_id] = {"id": new_id, "name": student.name, "grade": student.grade}
    return students[new_id]


# =============================================================================
# STEP 6 — PUT: update something
# =============================================================================
# WHAT IS PUT?
#   PUT means "replace the object that has this id with the data I am sending".
#   It uses BOTH: the id comes from the path, the new values from the body.
#
# POST vs PUT — the question students always ask:
#   POST /students     -> creates a NEW student, the server picks the id
#   PUT  /students/1   -> overwrites the EXISTING student number 1
#   Sending the same POST twice creates two students. Sending the same PUT
#   twice leaves exactly the same result — that is why PUT is "safe to repeat".
#
# TEST IT:
#   curl -X PUT http://127.0.0.1:8000/students/1 \
#        -H "Content-Type: application/json" \
#        -d '{"name": "Ali", "grade": 95}'
#
#   check:  curl http://127.0.0.1:8000/students/1     -> grade is now 95
#   try a missing id: curl -X PUT http://127.0.0.1:8000/students/99 \
#        -H "Content-Type: application/json" -d '{"name": "X", "grade": 50}'


@app.put("/students/{student_id}")
def update_student(student_id: int, student: Student):
    if student_id not in students:
        return {"error": "student not found"}
    students[student_id] = {"id": student_id, "name": student.name, "grade": student.grade}
    return students[student_id]


# =============================================================================
# STEP 7 — DELETE: remove something
# =============================================================================
# WHAT IS DELETE?
#   DELETE removes the object with that id. It needs only the path — no body.
#
# TELL THE STUDENTS:
#   * Returning the deleted object is a nice habit: the client can show
#     "Zara was deleted" or offer an undo.
#   * Deleting the same id twice: the first call works, the second reports
#     "not found". Always handle the missing case.
#
# TEST IT:
#   curl -X DELETE http://127.0.0.1:8000/students/2
#   curl -X DELETE http://127.0.0.1:8000/students/2    (second time: not found)
#   curl http://127.0.0.1:8000/students                (Zara is gone)


@app.delete("/students/{student_id}")
def delete_student(student_id: int):
    if student_id not in students:
        return {"error": "student not found"}
    deleted = students.pop(student_id)
    return {"deleted": deleted}


# =============================================================================
# SUMMARY — put this on the board
# =============================================================================
#   METHOD    DECORATOR       MEANING          WHERE IS THE DATA?
#   GET       @app.get        read             URL only
#   POST      @app.post       create           body (JSON)
#   PUT       @app.put        update/replace   id in URL + body
#   DELETE    @app.delete     delete           URL only
#
#   /students/{id}   path parameter   -> WHICH object
#   ?name=Ali        query parameter  -> options, filters, search
#   BaseModel        request body     -> the full object (POST / PUT)
#
# HOMEWORK IDEAS:
#   1. Add GET /students/{id}/passed returning {"passed": grade >= 60}.
#   2. Add a query filter: GET /students?min_grade=90.
#   3. Add a `teachers` dict and write all four methods for it.


# =============================================================================
# Demo — run this file (no server needed) to see every request and answer
# =============================================================================


def demo():
    from fastapi.testclient import TestClient

    client = TestClient(app)

    def show(label, response):
        print(f"\n{label}\n  -> {response.status_code}  {response.json()}")

    print("\n--- STEP 1: GET ---")
    show("GET  /", client.get("/"))

    print("\n--- STEP 2: path parameter ---")
    show("GET  /hello/Ali", client.get("/hello/Ali"))
    show("GET  /age/20", client.get("/age/20"))
    show("GET  /age/abc   (error: not a number)", client.get("/age/abc"))

    print("\n--- STEP 3: query parameter ---")
    show("GET  /greet?name=Ali", client.get("/greet?name=Ali"))
    show("GET  /greet?name=Ali&loud=true", client.get("/greet?name=Ali&loud=true"))
    show("GET  /greet   (error: name is required)", client.get("/greet"))

    print("\n--- STEP 4: read the data ---")
    show("GET  /students", client.get("/students"))
    show("GET  /students/1", client.get("/students/1"))
    show("GET  /students/99", client.get("/students/99"))

    print("\n--- STEP 5: POST (create) ---")
    show("POST /students", client.post("/students", json={"name": "Bek", "grade": 70}))
    show("POST /students  (error: grade is not a number)",
         client.post("/students", json={"name": "Bek", "grade": "excellent"}))

    print("\n--- STEP 6: PUT (update) ---")
    show("PUT  /students/1", client.put("/students/1", json={"name": "Ali", "grade": 95}))

    print("\n--- STEP 7: DELETE ---")
    show("DELETE /students/2", client.delete("/students/2"))
    show("DELETE /students/2  (again: already gone)", client.delete("/students/2"))

    show("GET  /students   (final state)", client.get("/students"))

    print("\nNow run:  uvicorn fastapi_lesson:app --reload")
    print("and open: http://127.0.0.1:8000/docs\n")


if __name__ == "__main__":
    demo()
