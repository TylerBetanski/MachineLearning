# Music Classifier (Country vs. Hip-Hop)

Goal: Create a classifier to distinguish between Country and Hip-Hop songs, based on their attributes.

### Features:
><b>Danceability:</b> Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable\
>\
><b>Energy:</b> Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale\
>\
><b>Speechiness:</b> Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks\
>\
><b>Acousticness:</b> A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic\
>\
><b>Valence:</b> A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)\
>\
><b>Tempo:</b> The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration

### Label:
><b>Genre:</b> The genre of the music (For our purposes, either Country or Hip-Hop)

Country adn Hip-Hop were selected to distinguish between because they're fairly opposite genres, and they should lead to a nice seperation of data.

# Results:
Initially a Support Vector machine was used as the classifier, resulting in a training accuracy of around 92% and a validation accuracy of 90%.\
<img src="https://github.com/user-attachments/assets/f39d7294-e2b5-4bc0-85b9-9f0052a84a26" width="400"></img>\
Looking at the Confusion Matrix (where 0 is Country and 1 is Hip-Hop), it's significantly more likely for Hip-Hop to be confused for country than for country to be for Hip-Hop. This could be because the Hip-Hop genre is more experimental and is always evolving (making it harder to classify), while Country music has been pretty stagnant for the past 15 years.


Next I created a Neural Network to perform the same classification, to test the performance and efficiency of a Neural Network against Support Vector Machines.

The Network is very shallow, consisting of 3 hidden layers, taking the data from 6 → 36 → 250 → 4 → 2 features. Each layer is followed with a ReLu activation function. A dropout with p=0.2 is used after the first hidden layer to drop 20% of the data.

The network performed with a similar accuracy to the SVC, with an average accuracy of 90 for both the Test & Training sets.\
<img src="https://github.com/user-attachments/assets/4b98d8fc-45fa-4735-874b-72c9f54c6567" width="400"></img>

As with the SVC, Hip-Hop is more often misidentified than Country.

To continue this project, I think it would be interesting to use the full list of features from the dataset (excluding identifying fields) and to use PCA or an Autoencoder to see which features are the most important to maintain adequate separation between different genres. I'd imagine that certain pairs of genres could be easily distinguished by just looking at 2 or 3 features (imagine Classical Opera vs. Death Metal).

# References:
Dataset sourced from: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
