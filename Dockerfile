FROM python:3.12

WORKDIR /therapist

# Copy only the requirements.txt first
COPY requirements.txt /therapist/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application files
COPY . /therapist/

EXPOSE 8080

CMD ["python", "app.py"]
