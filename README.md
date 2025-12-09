```bash
python manage.py makemigrations
python manage.py migrate
git add */migrations/*.py
git commit -m "Apply Collaboration Hub migrations"

git add collaboration/templates/
git commit -m "Add Collaboration Hub templates"

git add ml_playground/settings.py ml_playground/urls.py
git commit -m "Register Collaboration Hub in settings and URLs"
```