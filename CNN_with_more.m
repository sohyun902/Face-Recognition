 %newCNNwithlargerimagesizeandmorelayerandmoredataaugmentation
%Definethepath to the LFWdatasetfolder
 lfw_folder = 'C:\Users\00\Desktop\lfw';
 %Step1:LoadandPreprocessData(Assigning Labels by FolderNames)
 datasetPath = 'C:\Users\00\Desktop\lfw'; % Replace with your dataset path
 imds =imageDatastore(datasetPath, ...
 'IncludeSubfolders', true, ...
 'LabelSource', 'foldernames'); % Label each image with the name of its parent folder
 %Countthenumberofimagesforeachlabel (folder name represents theperson)
 labelCounts = countEachLabel(imds);
 %Filter labels with at least 30 images
 minImagesPerLabel = 40; % Adjust this to 40 ifneeded
 labelsToKeep = labelCounts(labelCounts.Count >= minImagesPerLabel, :);
 %Displaythefiltered labels and their counts
 disp(labelsToKeep);
 %Filter the image datastore to include only the filtered labels
 imds =subset(imds, ismember(imds.Labels, labelsToKeep.Label));
 %Shuffle the entire dataset to ensure randomization
 imds =shuffle(imds);
 %Select 30%ofthedataset fortesting
 numFiles =numel(imds.Files);
 numTestFiles = round(0.3 * numFiles); % 30%for testing
 %Randomlyselectindices for testing
 testIdx = randperm(numFiles, numTestFiles);
 trainIdx = setdiff(1:numFiles, testIdx);
 %Split dataset into training (70%) and testing (30% subset)
 imdsTrain = subset(imds, trainIdx); % 70% data for training
imdsTest = subset(imds, testIdx); % 30% subset for testing
 %Resizeimagestoasmaller size for faster training
 inputSize = [224 224 3]; %Increaseimage size for better feature extraction
 augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
 'DataAugmentation', imageDataAugmenter( ...
 'RandRotation', [-20, 20], ... % Increase the rotation range
 'RandXTranslation', [-10 10], ...
 'RandYTranslation', [-10 10], ...
 'RandXScale', [0.9 1.1], ... % Scale images
 'RandYScale', [0.9 1.1], ...
 'RandXShear', [-5 5], ...
 'RandYShear', [-5 5]));
 augimdsTest = augmentedImageDatastore(inputSize, imdsTest);
 %Step2:Simplified CNNArchitecture
 layers = [
 imageInputLayer(inputSize)
 convolution2dLayer(3, 32, 'Padding', 'same')
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2, 'Stride', 2)
 convolution2dLayer(3, 64, 'Padding', 'same')
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2, 'Stride', 2)
 convolution2dLayer(3, 128, 'Padding', 'same')
 batchNormalizationLayer
reluLayer
 maxPooling2dLayer(2, 'Stride', 2)
 %Addmoreconvolutional layers
 convolution2dLayer(3, 256, 'Padding', 'same')
 batchNormalizationLayer
 reluLayer
 maxPooling2dLayer(2, 'Stride', 2)
 fullyConnectedLayer(512) % Increase the size of fully connected layers
 reluLayer
 dropoutLayer(0.5) % Add dropout to reduceoverfitting
 fullyConnectedLayer(numel(categories(imdsTrain.Labels))) % Output layer
 softmaxLayer
 classificationLayer];
 %Step3:Training Options
 options = trainingOptions('sgdm', ...
 'InitialLearnRate', 0.001, ... % Slightly higher learning rate
 'MaxEpochs', 15, ... %Train for longer
 'MiniBatchSize', 32, ... % Smaller batch size for better updates
 'ValidationData', augimdsTest, ...
 'ValidationFrequency', 30, ...
 'Verbose', false, ...
 'Plots', 'training-progress');
 %Step4:TraintheCNN
 net = trainNetwork(augimdsTrain, layers, options);
 %Step5:Evaluate theModel
 YPred =classify(net, augimdsTest); % Predict the labels for the test set
YTest =imdsTest.Labels; % True labels
 %Calculate accuracy
 accuracy = sum(YPred ==YTest) / numel(YTest);
 disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);
 %TestAccuracy: 57.9934%
 %TestAccuracy: 63.4318%
 %TestAccuracy: 75.1786%