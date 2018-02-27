FROM skorch:latest

WORKDIR /app
RUN pip3 install -r requirements-dev.txt

# For seq2seq translation example (translation.ipynb)
ADD http://www.manythings.org/anki/fra-eng.zip /app/examples/translation/data/

RUN unzip /app/examples/translation/data/fra-eng.zip -d /app/examples/translation/data/
RUN mv /app/examples/translation/data/fra.txt /app/examples/translation/data/eng-fra.txt

# Expose jupyter notebook port (installed via requirements-dev.txt)
EXPOSE 8888
