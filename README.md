# Nepali Text Summarizer

A Django-based web application that provides text summarization for Nepali content using TextRank algorithm and word embeddings.

## Features

- Extractive text summarization for Nepali text
- Support for financial and numerical content
- Rule-based Nepali stemmer
- Custom word embeddings for better Nepali language understanding

## Requirements

- Python 3.8+
- Django 5.1.1
- NumPy
- scikit-learn
- NetworkX
- Gunicorn
- WhiteNoise

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd nepali_summarizer
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run migrations:
```bash
python manage.py migrate
```

5. Start the development server:
```bash
python manage.py runserver
```

## Deployment

The application is configured for deployment on Render. The following environment variables should be set:

- `SECRET_KEY`: Django secret key
- `DEBUG`: Set to 'False' in production

## License

This project is licensed under the MIT License. 