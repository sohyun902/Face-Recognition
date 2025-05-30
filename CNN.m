 %oldCNN70/30training andtesting
 %Definethepath to the LFWdatasetfolder
 lfw_folder = 'C:\Users\00\Desktop\lfw';
 %Step1:LoadandPreprocessData(Assigning Labels by FolderNames)
 datasetPath = 'C:\Users\00\Desktop\lfw'; % Replace with your dataset path
 imds = imageDatastore(datasetPath, ...
 'IncludeSubfolders', true, ...
 'LabelSource', 'foldernames'); % Label each image with the name of its parent folder
 %Countthenumberofimagesforeachlabel (folder name represents theperson)
 labelCounts = countEachLabel(imds);
 %Filter labels with at least 20/30/40 images
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
 inputSize = [128 128 3]; %Smaller image size forfaster training
 augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
 'DataAugmentation', imageDataAugmenter( ...
 'RandRotation', [-15, 15], ... % Smaller rotation
 'RandXTranslation', [-5 5], ...
 'RandYTranslation', [-5 5]));
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
fullyConnectedLayer(256)
 reluLayer
 fullyConnectedLayer(numel(categories(imdsTrain.Labels))) % Number of output classes
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
 %TestAccuracy: 65.1982%-> minof20imageperperson
 %TestAccuracy: 70.1625%-> minof30imageperperson
 %TestAccuracy: 80.1786%-> minof40imageperperson