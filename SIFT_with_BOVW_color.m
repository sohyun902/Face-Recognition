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