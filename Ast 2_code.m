 %oldCNN70/30training andtesting
 %Definethepath to the LFWdatasetfolder
 lfw_folder = 'C:\Users\00\Desktop\lfw';
 %Step1:LoadandPreprocessData(Assigning Labels by FolderNames)
 datasetPath = 'C:\Users\00\Desktop\lfw'; % Replace with your dataset path
 imds =imageDatastore(datasetPath, ...
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

 %CNNwith70/30training/testing data
 %Definethepath to the LFWdatasetfolder
 lfw_folder = 'C:\Users\00\Desktop\lfw';
 %Step1:LoadandPreprocessData(Assigning Labels by FolderNames)
 datasetPath = 'C:\Users\00\Desktop\lfw'; % Replace with your dataset path
 imds =imageDatastore(datasetPath, ...
 'IncludeSubfolders', true, ...
 'LabelSource', 'foldernames'); % Label each image with the name of its parent folder
 %Countthenumberofimagesforeachlabel (folder name represents theperson)
 labelCounts = countEachLabel(imds);
 %Filter labels with at least 40 images
 minImagesPerLabel = 40; % Adjust this to 40 ifneeded
 labelsToKeep = labelCounts(labelCounts.Count >= minImagesPerLabel, :);
 %Displaythefiltered labels and their counts
 disp(labelsToKeep);
 %Filter the image datastore to include only the filtered labels
 imds =subset(imds, ismember(imds.Labels, labelsToKeep.Label));
 %Shuffle the entire dataset to ensure randomization
 imds =shuffle(imds);
%Select 70%ofthedataset fortraining
 numFiles =numel(imds.Files);
 numTrainFiles = round(0.9 * numFiles); % 70/80/90%for training
 %Randomlyselectindices for training
 trainIdx = randperm(numFiles, numTrainFiles);
 testIdx = setdiff(1:numFiles, trainIdx);
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
 %TestAccuracy: 81.4286%-> 70/30
 %TestAccuracy: 79.0885%-> 80/20
 %TestAccuracy: 82.8877%-> 90/10

 %SIFTfeature gray color
 %Replacewiththe pathto yourextracted VLFeat folder
 run('C:\Users\00\Desktop\vlfeat-0.9.21/toolbox/vl_setup');
 vl_version
 %Definethepath to the LFWdatasetfolder
 datasetPath = 'C:\Users\00\Desktop\lfw';
 imds =imageDatastore(datasetPath, ...
 'IncludeSubfolders', true, ...
 'LabelSource', 'foldernames'); % Label each image with the name of its parent folder
 %Countthenumberofimagesforeachlabel
 labelCounts = countEachLabel(imds);
 %Filter labels with at least 50 images
 minImagesPerLabel = 50;
 labelsToKeep = labelCounts(labelCounts.Count >= minImagesPerLabel, :);
 %Displaythefiltered labels and their counts as a bar chart
 figure;
 bar(categorical(labelsToKeep.Label), labelsToKeep.Count);
 title('Label Distribution After Filtering');
 xlabel('Labels');
 ylabel('Number of Images');
 %Filter the image datastore to include only the filtered labels
imds =subset(imds, ismember(imds.Labels, labelsToKeep.Label));
 %Initialize feature and label storage
 numImages=numel(imds.Files);
 features = [];
 labels = [];
 %Loopthrougheachimage,extract SIFT features, and store them
 for i = 1:numImages
 img=readimage(imds, i);
 imgGray =rgb2gray(img); % Convert tograyscale
 %Extract SIFT features (using VLFeat library or similar)
 [frames, descriptors] = vl_sift(single(imgGray));
 %Aggregate features and store them(you cantake the mean, forexample)
 features = [features; mean(descriptors, 2)'];
 %Storecorresponding label
 labels = [labels; imds.Labels(i)];
 end
 %Convertlabels to categorical
 labels = categorical(labels);
 %Visualize the SIFT keypoints on a single image
 sampleImage =readimage(imds, 1); % Takethe firstimage for visualization
 imgGray =rgb2gray(sampleImage); % Convertto grayscale
 %Extract SIFT features for visualization
 [frames, ~] = vl_sift(single(imgGray));
 %Displaythegrayscale image with keypoints overlaid
 figure;
 imshow(imgGray);
hold on;
 h=vl_plotframe(frames);
 set(h, 'color', 'y', 'linewidth', 2); % Plot keypoints in yellow
 title('SIFT Keypoints on Sample Image');
 hold off;
 %Split the dataset into training and testing sets (70% training, 30% testing)
 trainRatio = 0.7;
 [trainInd, testInd] = dividerand(numImages, trainRatio, 1- trainRatio);
 trainFeatures = features(trainInd, :);
 trainLabels = labels(trainInd);
 testFeatures = features(testInd, :);
 testLabels = labels(testInd);
 %Trainaclassifier (e.g., SVM)
 classifier = fitcecoc(trainFeatures, trainLabels);
 %Testtheclassifier
 predictedLabels = predict(classifier, testFeatures);
 %Calculate accuracy
 accuracy = sum(predictedLabels == testLabels) / numel(testLabels) * 100;
 fprintf('Accuracy: %.2f%%\n', accuracy);
 %Createaconfusion matrix to evaluate classifier performance
 figure;
 confusionchart(testLabels, predictedLabels);
 title('Confusion Matrix');
 %Accuracy: 32.68%

 %SIFTwithBAGOFvw
 %Initialize VLFeat toolbox (make sure it's added to your MATLAB path)
run('C:/path_to_vlfeat/toolbox/vl_setup');
 %Definethepath to the LFWdatasetfolder
 datasetPath = 'C:\Users\00\Desktop\lfw';
 imds =imageDatastore(datasetPath, ...
 'IncludeSubfolders', true, ...
 'LabelSource', 'foldernames'); % Label each image with the name of its parent folder
 %Countthenumberofimagesforeachlabel
 labelCounts = countEachLabel(imds);
 %Filter labels with at least 50 images
 minImagesPerLabel = 50;
 labelsToKeep = labelCounts(labelCounts.Count >= minImagesPerLabel, :);
 %Displaythefiltered labels and their counts
 disp(labelsToKeep);
 %Filter the image datastore to include only the filtered labels
 imds =subset(imds, ismember(imds.Labels, labelsToKeep.Label));
 %Initialize storage for all SIFT descriptors
 allDescriptors = [];
 %Extract SIFT features from all images
 numImages=numel(imds.Files);
 for i = 1:numImages
 img=readimage(imds, i);
 imgGray =single(rgb2gray(img)); % Convert to grayscale and single precision
 %Extract SIFT features using VLFeat
 [~, descriptors] = vl_sift(imgGray);
 %Storeall descriptors
 allDescriptors = [allDescriptors descriptors]; % Concatenate descriptors from all images
end
 %Step2:ClusterSIFT descriptors into 'visual words' using k-means
 numClusters = 100; %Adjustbasedondataset size andcomplexity
 [centers, ~] = vl_kmeans(single(allDescriptors), numClusters);
 %Visualize the first few cluster centers (visual words) as bar charts
 numWordsToShow=10;%Numberofvisualwordstovisualize
 figure;
 for i = 1:numWordsToShow
 subplot(2, 5, i); % Create a 2x5 grid for visualizing
 bar(centers(:, i)); % Each center is a 128-dimensional vector
 title(['Word ' num2str(i)]);
 xlabel('SIFT Descriptor Dimension');
 ylabel('Value');
 end
 sgtitle('Visual Words (Cluster Centers as SIFT Descriptors)');
 %Step3:Create histograms of visual words foreach image
 features = zeros(numImages, numClusters); % Store histograms for each image
 for i = 1:numImages
 img=readimage(imds, i);
 imgGray =single(rgb2gray(img)); % Convert to grayscale and single precision
 %Extract SIFT features for the current image
 [~, descriptors] = vl_sift(imgGray);
 %Assigneachdescriptor to the closest visual word
 assignments = vl_kdtreequery(vl_kdtreebuild(centers), centers, single(descriptors));
 %Buildhistogramof visual words
 histo = histcounts(assignments, 1:numClusters+1);
%Storehistogramasfeatures for this image
 features(i, :) = histo;
 end
 %Normalizehistograms to get a uniformscale (optional)
 features = normalize(features, 2, 'norm', 1);
 %Visualize the histogram of visual words for a sample image
 sampleImageIndex= 1; %Changethisindexto visualize other images
 histo = features(sampleImageIndex, :); % Histogram of visual words for this image
 figure;
 bar(histo);
 title(['Histogram of Visual Words for Image ' num2str(sampleImageIndex)]);
 xlabel('Visual Word Index');
 ylabel('Frequency');
 %Convertlabels to categorical for classification
 labels = categorical(imds.Labels);
 %Step4:Split the data into training and testing sets (70% training, 30% testing)
 trainRatio = 0.7;
 [trainInd, testInd] = dividerand(numImages, trainRatio, 1- trainRatio);
 trainFeatures = features(trainInd, :);
 trainLabels = labels(trainInd);
 testFeatures = features(testInd, :);
 testLabels = labels(testInd);
 %Step5:Trainaclassifier (SVM in this case)
 classifier = fitcecoc(trainFeatures, trainLabels);
 %Step6:Testtheclassifier
 predictedLabels = predict(classifier, testFeatures);
%Calculate accuracy
 accuracy = sum(predictedLabels == testLabels) / numel(testLabels) * 100;
 fprintf('Bag of Visual Words Accuracy: %.2f%%\n', accuracy);
 %Confusionmatrix visualization
 figure;
 cm=confusionchart(testLabels, predictedLabels);
 title('Confusion Matrix');
 %BagofVisual WordsAccuracy: 33.66%
 %Loopthroughthetestset and find successful recognitions
 correctIndices = find(predictedLabels == testLabels); % Indices of correctly classified
 images
 incorrectIndices = find(predictedLabels ~= testLabels); % Indices of incorrectly classified
 images
 %Displayafewsuccessful recognitions
 numToShow=5;%Numberofexamplestoshow
 figure;
 sgtitle('Successful Recognitions');
 for i = 1:min(numToShow, numel(correctIndices))
 idx = correctIndices(i);
 img=readimage(imds, testInd(idx)); % Read the image from the testset
 subplot(1, numToShow, i);
 imshow(img);
 title(['True: ' char(testLabels(idx)) ', Predicted: ' char(predictedLabels(idx))]);
 end
 %Displayafewincorrect recognitions
 figure;
 sgtitle('Misclassified Images');
for i = 1:min(numToShow, numel(incorrectIndices))
 idx = incorrectIndices(i);
 img=readimage(imds, testInd(idx)); % Read the image from the testset
 subplot(1, numToShow, i);
 imshow(img);
 title(['True: ' char(testLabels(idx)) ', Predicted: ' char(predictedLabels(idx))]);
end

 %SIFTwithcolor conversion and bag of words
 %Initialize VLFeat toolbox (make sure it's added to your MATLAB path)
 run('C:/path_to_vlfeat/toolbox/vl_setup');
 %Definethepath to the LFWdatasetfolder
 datasetPath = 'C:\Users\00\Desktop\lfw';
 imds =imageDatastore(datasetPath, ...
 'IncludeSubfolders', true, ...
 'LabelSource', 'foldernames'); % Label each image with the name of its parent folder
 %Countthenumberofimagesforeachlabel
 labelCounts = countEachLabel(imds);
 %Filter labels with at least 50 images
 minImagesPerLabel = 50;
 labelsToKeep = labelCounts(labelCounts.Count >= minImagesPerLabel, :);
 %Displaythefiltered labels and their counts
 disp(labelsToKeep);
 %Filter the image datastore to include only the filtered labels
 imds =subset(imds, ismember(imds.Labels, labelsToKeep.Label));
 %Initialize storage for all SIFT descriptors from all images
 allDescriptors = [];
%Extract SIFT features from all images in each RGB channel
 numImages=numel(imds.Files);
 for i = 1:numImages
 img=readimage(imds, i);
 %Split imageinto Red, Green, and Bluechannels
 redChannel = single(img(:, :, 1)); % Red channel
 greenChannel = single(img(:, :, 2)); % Green channel
 blueChannel = single(img(:, :, 3)); % Blue channel
 %ApplySIFTtoeachchannelseparately
 [~, descriptorsR] = vl_sift(redChannel);
 [~, descriptorsG] = vl_sift(greenChannel);
 [~, descriptorsB] = vl_sift(blueChannel);
 %Ensurethenumberofdescriptorsmatches across channels
 numDescriptorsR = size(descriptorsR, 2);
 numDescriptorsG = size(descriptorsG, 2);
 numDescriptorsB = size(descriptorsB, 2);
 %Taketheminimumnumberofkeypointsacrossallchannels
 minDescriptors = min([numDescriptorsR, numDescriptorsG, numDescriptorsB]);
 %Keeponlythefirst 'minDescriptors' descriptors from each channel
 descriptorsR = descriptorsR(:, 1:minDescriptors);
 descriptorsG = descriptorsG(:, 1:minDescriptors);
 descriptorsB = descriptorsB(:, 1:minDescriptors);
 %Concatenate descriptors from all channels horizontally (side-by-side)
 descriptors = [descriptorsR; descriptorsG; descriptorsB];
 %Storeconcatenated descriptors
 allDescriptors = [allDescriptors descriptors];
end
 %Step2:ClusterSIFT descriptors into 'visual words' using k-means
 numClusters = 100; %Adjustbasedondataset size andcomplexity
 [centers, ~] = vl_kmeans(single(allDescriptors), numClusters);
 %Step3:Create histograms of visual words foreach image
 features = zeros(numImages, numClusters); % Store histograms for each image
 for i = 1:numImages
 img=readimage(imds, i);
 %Split imageinto Red, Green, and Bluechannels
 redChannel = single(img(:, :, 1)); % Red channel
 greenChannel = single(img(:, :, 2)); % Green channel
 blueChannel = single(img(:, :, 3)); % Blue channel
 %ApplySIFTtoeachchannelseparately
 [~, descriptorsR] = vl_sift(redChannel);
 [~, descriptorsG] = vl_sift(greenChannel);
 [~, descriptorsB] = vl_sift(blueChannel);
 %Ensurethenumberofdescriptorsmatches across channels
 numDescriptorsR = size(descriptorsR, 2);
 numDescriptorsG = size(descriptorsG, 2);
 numDescriptorsB = size(descriptorsB, 2);
 %Taketheminimumnumberofkeypointsacrossallchannels
 minDescriptors = min([numDescriptorsR, numDescriptorsG, numDescriptorsB]);
 %Keeponlythefirst 'minDescriptors' descriptors from each channel
 descriptorsR = descriptorsR(:, 1:minDescriptors);
 descriptorsG = descriptorsG(:, 1:minDescriptors);
 descriptorsB = descriptorsB(:, 1:minDescriptors);
%Concatenate descriptors from all channels horizontally
 descriptors = [descriptorsR; descriptorsG; descriptorsB];
 %Assigneachdescriptor to the closest visual word
 assignments = vl_kdtreequery(vl_kdtreebuild(centers), centers, single(descriptors));
 %Buildhistogramof visual words
 histo = histcounts(assignments, 1:numClusters+1);
 %Storehistogramasfeatures for this image
 features(i, :) = histo;
 end
 %Normalizehistograms to get a uniformscale (optional)
 features = normalize(features, 2, 'norm', 1);
 %Convertlabels to categorical for classification
 labels = categorical(imds.Labels);
 %Step4:Split the data into training and testing sets (70% training, 30% testing)
 trainRatio = 0.7;
 [trainInd, testInd] = dividerand(numImages, trainRatio, 1- trainRatio);
 trainFeatures = features(trainInd, :);
 trainLabels = labels(trainInd);
 testFeatures = features(testInd, :);
 testLabels = labels(testInd);
 %Step5:Trainaclassifier (SVM in this case)
 classifier = fitcecoc(trainFeatures, trainLabels);
 %Step6:Testtheclassifier
 predictedLabels = predict(classifier, testFeatures);
 %Calculate accuracy
 accuracy = sum(predictedLabels == testLabels) / numel(testLabels) * 100;
fprintf('Color-SIFT (Bag of Visual Words) Accuracy: %.2f%%\n', accuracy);
 %Confusionmatrix visualization
 figure;
 cm=confusionchart(testLabels, predictedLabels);
 title('Confusion Matrix');
 %Color-SIFT (Bag of Visual Words)Accuracy: 35.63%
 %Findcorrect and incorrect classifications
 correctIndices = find(predictedLabels == testLabels);
 incorrectIndices = find(predictedLabels ~= testLabels);
 numToShow=2;%Sethowmanyexamplestoshow
 %Visualization for Correct Classifications
 figure;
 sgtitle('Successful Predictions with Visual Word Histograms');
 for i = 1:min(numToShow, numel(correctIndices))
 idx = correctIndices(i);
 %Displaythecorrectly classified image
 img=readimage(imds, testInd(idx)); % Read the image
 subplot(2, numToShow, i);
 imshow(img);
 title(['True: ' char(testLabels(idx)) ', Predicted: ' char(predictedLabels(idx))]);
 %Displaythehistogram of visual words for this image
 subplot(2, numToShow, i +numToShow);
 bar(trainFeatures(testInd(idx), :));
 title('Visual Words Histogram');
 xlabel('Visual Word Index');
 ylabel('Frequency');
end
 %Visualization for Incorrect Classifications
 figure;
 sgtitle('Misclassified Images with Visual Word Histograms');
 for i = 1:min(numToShow, numel(incorrectIndices))
 idx = incorrectIndices(i);
 %Displaytheincorrectly classified image
 img=readimage(imds, testInd(idx)); % Read the image
 subplot(2, numToShow, i);
 imshow(img);
 title(['True: ' char(testLabels(idx)) ', Predicted: ' char(predictedLabels(idx))]);
 %Displaythehistogram of visual words for this image
 subplot(2, numToShow, i +numToShow);
 bar(trainFeatures(testInd(idx), :));
 title('Visual Words Histogram');
 xlabel('Visual Word Index');
 ylabel('Frequency');
 end
 %Step1:LoadandPreprocessData
 datasetPath = '/MATLAB Drive/.Trash-1000600715/aname'; % Replace with your dataset
 path
 imds =imageDatastore(datasetPath, ...
 'IncludeSubfolders', true, ...
 'LabelSource', 'foldernames');
 %Split dataset into training (100%) and testing (30%)
 imds =shuffle(imds); % Shuffle the dataset
numFiles =numel(imds.Files);
 numTestFiles = round(0.3 * numFiles); % 30%for testing
 testIdx = randperm(numFiles, numTestFiles);
 trainIdx = setdiff(1:numFiles, testIdx);
 imdsTrain = subset(imds, trainIdx); % Training on all images
 imdsTest = subset(imds, testIdx); % 30% subset for testing
 %Step2:NoiseReduction (Gaussian Filter)
 img=readimage(imdsTrain, 1); % Load thefirst image from the training set
 if size(img, 3) == 3
 imgGray =rgb2gray(img); % Convert tograyscale if RGB
 else
 imgGray =img;
 end
 imgFiltered = imgaussfilt(imgGray, 1); % Noise reduction using Gaussian filter
 %Step3:Scale-Space Pyramid &Difference of Gaussian (DoG)
 octaves = 4; %Numberofoctaves
 sigma =1.6; %Initial Gaussian sigma
 k =sqrt(2); % Scaling factor
 scaleSpacePyramid = cell(octaves, 1);
 dogPyramid= cell(octaves- 1, 1); % Difference of Gaussian pyramid
 for i = 1:octaves
 gaussImg1 =imgaussfilt(imgFiltered, sigma * (k^(i-1)));
 gaussImg2 =imgaussfilt(imgFiltered, sigma * (k^i));
 scaleSpacePyramid{i} = gaussImg1;
 dogPyramid{i} = gaussImg2- gaussImg1;
 end
%DisplayoneDoGimage
 figure;
 imshow(dogPyramid{2}, []);
 title('Difference of Gaussian');
 %Step4:Keypoint Localization (Finding Local Maxima/Minima inDoG)
 keypoints = cell(size(dogPyramid));
 for i = 1:length(dogPyramid)
 keypoints{i} = imregionalmax(dogPyramid{i}); % Find local maxima
 end
 %Displaykeypoints on theoriginal image
 figure;
 imshow(imgGray);
 hold on;
 for i = 1:length(keypoints)
 [y, x] = find(keypoints{i});
 plot(x, y, 'ro'); % Mark keypoints
 end
 title('Detected Keypoints');
 hold off;
 %Step5:OrientationAssignment (Based on Gradient)
 [gradientX, gradientY] = imgradientxy(imgFiltered);
 [magnitude, orientation] = imgradient(gradientX, gradientY);
 figure;
 imshow(imgFiltered);
 hold on;
 for i = 1:length(keypoints)
[y, x] = find(keypoints{i});
 for j = 1:length(x)
 plot(x(j), y(j), 'ro');
 quiver(x(j), y(j), cosd(orientation(y(j), x(j))), sind(orientation(y(j), x(j))), 'g');
 end
 end
 title('Keypoints with Orientation Assignment');
 hold off;
 %Step1:LoadandPreprocessData
 datasetPath = '/MATLAB Drive/.Trash-1000600715/aname'; % Replace with your dataset
 path
 imds =imageDatastore(datasetPath, ...
 'IncludeSubfolders', true, ...
 'LabelSource', 'foldernames');
 %Split dataset into training (70%) and testing (30%)
 imds =shuffle(imds); % Shuffle the dataset
 numFiles =numel(imds.Files);
 numTestFiles = round(0.3 * numFiles); % 30%for testing
 testIdx = randperm(numFiles, numTestFiles);
 trainIdx = setdiff(1:numFiles, testIdx);
 imdsTrain = subset(imds, trainIdx); % Training on all images
 imdsTest = subset(imds, testIdx); % 30% subset for testing
 %Step2:NoiseReduction (Gaussian Filter)
 img1 =readimage(imdsTrain, 1); % Loadthe first image from the training set
 img2 =readimage(imdsTrain, 2); % Loadthe secondimage for comparison
 %Converttograyscale
if size(img1, 3) == 3
 imgGray1= rgb2gray(img1);
 else
 imgGray1= img1;
 end
 if size(img2, 3) == 3
 imgGray2= rgb2gray(img2);
 else
 imgGray2= img2;
 end
 %Noisereduction using Gaussian filter
 imgFiltered1 = imgaussfilt(imgGray1, 1);
 imgFiltered2 = imgaussfilt(imgGray2, 1);
 %Step3:Scale-Space Pyramid &Difference of Gaussian (DoG)
 octaves = 4; %Numberofoctaves
 sigma =1.6; %Initial Gaussian sigma
 k =sqrt(2); % Scaling factor
 %Applyonthefirst image
 scaleSpacePyramid1 = cell(octaves, 1);
 dogPyramid1= cell(octaves- 1, 1);
 for i = 1:octaves
 gaussImg1 =imgaussfilt(imgFiltered1, sigma * (k^(i-1)));
 gaussImg2 =imgaussfilt(imgFiltered1, sigma * (k^i));
 scaleSpacePyramid1{i} = gaussImg1;
 dogPyramid1{i} = gaussImg2- gaussImg1;
 end
%Applyonthesecondimage
 scaleSpacePyramid2 = cell(octaves, 1);
 dogPyramid2= cell(octaves- 1, 1);
 for i = 1:octaves
 gaussImg1 =imgaussfilt(imgFiltered2, sigma * (k^(i-1)));
 gaussImg2 =imgaussfilt(imgFiltered2, sigma * (k^i));
 scaleSpacePyramid2{i} = gaussImg1;
 dogPyramid2{i} = gaussImg2- gaussImg1;
 end
 %Step4:Keypoint Localization (Finding Local Maxima/Minima inDoG)
 keypoints1 = cell(size(dogPyramid1));
 for i = 1:length(dogPyramid1)
 keypoints1{i} = imregionalmax(dogPyramid1{i}); % Find local maxima for the first image
 end
 keypoints2 = cell(size(dogPyramid2));
 for i = 1:length(dogPyramid2)
 keypoints2{i} = imregionalmax(dogPyramid2{i}); % Find local maxima for the second
 image
 end
 %Step5:Extracting Features and Matching Keypoints
 %DetectSIFTfeatures (use 'detectSURFFeatures' for SURF, or other detectors)
 features1 = detectSURFFeatures(imgFiltered1);
 features2 = detectSURFFeatures(imgFiltered2);
 [descriptors1, validPoints1] = extractFeatures(imgFiltered1, features1);
 [descriptors2, validPoints2] = extractFeatures(imgFiltered2, features2);
 %Matchfeaturesusing nearest neighbor search
indexPairs = matchFeatures(descriptors1, descriptors2);
 %Retrievematched points
 matchedPoints1 = validPoints1(indexPairs(:, 1), :);
 matchedPoints2 = validPoints2(indexPairs(:, 2), :);
 %Step6:Calculate Accuracy
 %Accuracycalculation based on number of correct matches
 numCorrectMatches = size(matchedPoints1, 1); % Numberof correctmatches
 totalKeypoints = size(validPoints1, 1); % Total keypoints in the first image
 accuracy = (numCorrectMatches / totalKeypoints) * 100;
 %Displayaccuracy
 fprintf('Accuracy of Keypoint Matching: %.2f%%\n', accuracy);
 %Step7:Visualize the matched keypoints
 figure;
 showMatchedFeatures(imgFiltered1, imgFiltered2, matchedPoints1, matchedPoints2,'montage');
 title('Matched Keypoints between Two Images');