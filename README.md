# Repository for skripsi

This project is about sentiment analysis on MyIndiHome user reviews. Built using tensorflow and flask

## To start (with cmd)
1. Clone this repository
```
git clone https://github.com/shirousaberx/skripsi
```

2. Move to the project directory
```
cd skripsi
```

3. Unpack the model
```
tar -xf model.zip
```

3. Create a new virtual environment
```
python -m venv env
```

5. Activate the virtual environment 
```
env\Scripts\activate
```

6. Install all dependencies
```
pip install -r requirements.txt
```

7. Run
```
flask run
```

8. Open browser and type in
```
localhost:5000
```

## Model Training Notebook
Notebook to train the model can be found at [model_training.ipynb](model_training.ipynb)
