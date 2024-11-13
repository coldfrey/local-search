# Real-Time Embedding Search
This repository provides a simple web application that allows you to perform real-time searches over a collection of notes using sentence embeddings. The application features a Flask backend and a responsive HTML frontend where you can search for notes, view a list of all notes, and click on any note to see its content.

## Demo Video
You can download and watch the demo video to see the application in action:

<video width="600" controls>
  <source src="/demo.mp4" type="video/mp4">
</video>

Features
Real-Time Search: Type into the search box to find the most similar note based on semantic similarity.
Notes List: View all your notes listed on the right-hand side of the screen.
View Note Content: Click on any note title to display its content.
Prerequisites
Before you begin, ensure you have met the following requirements:

Python 3.6 or higher is installed on your machine.
pip is available for installing Python packages.
Internet access is available for downloading model weights the first time.
Installation
1. Clone the Repository

```bash
git clone https://github.com/coldfrey/local-search.git
cd local-search
```
2. Install Required Python Packages
Install the required packages using pip:

``` bash
pip install -r requirements.txt
```

3. Prepare the Notes and Embeddings
Notes Directory: Place your .txt note files in the notes/ directory.

Generate Embeddings: Run the embeddings script to generate embeddings for your notes:

```bash
python embeddings.py
```
Running the Application
1. Start the Flask Backend
Run the main application script:

``` bash
python main.py
```
This will start the Flask server on http://127.0.0.1:5000/.

2. Open the Frontend
Open index.html in your web browser to access the application.


Adding New Notes
1. Add Your Note Files
Place any new .txt files into the notes/ directory.

2. Generate Embeddings for New Notes
Run the embeddings script again to process the new notes:

``` bash
python embeddings.py
```
This will generate corresponding .npy embedding files in the embeddings/ directory for the new notes.
