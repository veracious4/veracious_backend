# veracious_backend
Backend repository for Veracious project

 ## 1) Installing dependencies
### 1.1) Installing python 3.8
### 1.2) Installing Selenium webdriver for Chrome: https://chromedriver.chromium.org/downloads
### 1.3) Set up the os environment variable for the webdriver such as:
```bash
    SET SELENIUM_WEB_DRIVER_CHROME_PATH=<PATH_TO_CHROME_DRIVER>
```

## 2) Setting up project
### To create virtual environment
```
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

 ### To run fastapi server
```bash
cd ml_models/models_server
```

```bash
uvicorn main:app --reload
```
