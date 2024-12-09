Take into consideration, 


Launch app.py to set up local server, then launch run.py to use the program accordingly


For a more advanced approach:

Open your development environtment application and run the programs:

only run feed.py for classification, paste the video directory when asked, then it will be classified with the valid conclusions

but if you want to train the model yourself

have a set of .npy files ready, features of each videos

this can be prepared by a very large video dataset, have a directory called 'test',be ready with the videos, then run feature_extract.py

The features from each video (33 landmarks with 4 features each), needs to be statistically summarized to perform clustering using unsup.py

Then run super.py to train model using random forest.

due to upload limit, i cannot upload .npy files and label set
only the ready model, scaler and a sample video

create folders test, output_features and label beforehand as a precaution
