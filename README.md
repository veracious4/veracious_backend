# veracious_backend
Backend repository for Veracious project



### To create virtual environment
    
    ```bash
    python -m venv <ENVIRONMENT_NAME>
    ```
    for e.g.
    ```bash
    python -m venv backend_env
    ```

### To activate virtual environment

    ```bash
    <ENVIRONMENT_NAME>\Scripts\activate
    ``` 

### To install dependencies

    ```bash
    pip install -r requirements.txt
    ```

### To create requirements.txt

    ```bash
    pip freeze > requirements.txt
    ```

### To sync databases you can run
    ```bash
    cd backend_server
    ```
    
    ```bash
    python manage.py migrate
    ```

### To run the server
    ```bash
    cd backend_server
    ```

    ```bash
    python manage.py runserver
    ```
