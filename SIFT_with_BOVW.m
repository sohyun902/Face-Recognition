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