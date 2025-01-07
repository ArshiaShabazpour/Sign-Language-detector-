🎧 Sign Detector App Guide
Disclaimer: The latest commit is not fully functional as the app is transitioning to a Long Short-Term Memory (LSTM) model for more complex gesture recognition. The app currently supports detecting hand signs for the letters A, B, and D in American Sign Language (ASL).

📁 Table of Contents
🛠 Setup and Compilation
▶️ Running the Sign Detector
📲 Training and Data Collection
📂 Syntax Guide
🔗 Summary

🛠 Setup and Compilation
If you need to use the functional version of the app:

Clone the GitHub repository.

Refer to the first commit in the repository's history for the fully functional version.

Run the file data_collection.py to start the app. The app does not need to be run from the terminal.

Note: Changing dependencies may break functionality as the app is transitioning to an LSTM model.

▶️ Running the Sign Detector
To run the app and detect hand signs:

Open the app using data_collection.py.

Select detection mode from the menu by pressing 2.

The app will then use the SVM model to detect the signs A, B, and D from ASL.

📲 Training and Data Collection
To train and expand the app's sign recognition capabilities:

Run data_collection.py as described above.

Press 1 to select the training mode from the menu.

Enter the letter you want to add the sign for.

Capture frames for the selected letter using the following controls:

Space: Capture a frame.

Z: Delete the latest captured data.

Once enough data is captured, press S to save the captured data into the training dataset.

📂 Navigation Guide
This app uses a minimalistic control system:

1: Training mode.

2: Detection mode.

Space: Capture a frame.

Z: Delete the latest frame.

S: Save the captured frames.

