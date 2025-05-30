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