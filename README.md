# Data Collection and Managment Lab Final Project

## Description
This project used scraped event data to get the nearest events near AirBnb listings, as well as recommend a new price point and listing description during the event.

## Code Files
- *train_models_once.ipynb* : meant to be run on the DataBricks workspace. This file produces a listing-event pair table, as well as models for price prediction.
- *main_final_project.ipynb* : uploads the pair table as well as the learned models, and creates a new table including the price prediction, here is also an example of an end-user experience: enter listing id >  get near events > produce a new description using Gemini LLM model.
- *EventBnb.py* : a simple app that works on a sample of the data.

## Data Needed
- For the main notebook there is no need for any data files- all is in the DataBricks enviroment
- for the EventBnb.py file there needs to be the sample file (provided in the submissions) saved in the same spot where the event file exists.

## How to Use
### EventBnb.py:
- Open a terminal (cmd in windows search)
- Go to the location of the app (command: cd path)
- Install the following libraries: pip install pandas streamlit requests
- Run the app: streamlit run EventBnb.py
This will open the app on a browser where you can search up events using a listing id (some examples for ids are provided) and generate a new description

### main_final_project.ipynb:
- Open the notebook in the course's DataBricks enviroment
- Click "run all"
- enter a listing id as the input.

** Gemini note: You will be required to enter a gemini API code to generate text. This is free to get from https://ai.google.dev/

