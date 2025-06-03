c = parcluster('local');
c.NumWorkers = 90; % Set number of workers
saveProfile(c);

folderPath = "../data/derivatives/hctsa/BIS_prediction/tsmat";
outputFolder = '../data/derivatives/hctsa/BIS_prediction/hctsa';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

matFiles = dir(fullfile(folderPath, '*.mat'));
fileNames = {matFiles.name};
matFilePaths = fullfile(folderPath, fileNames);

% Filter out files that have already been processed
unprocessedFiles = {};
for i = 1:length(matFilePaths)
    inputFile = matFilePaths{i};
    [~, basename, ~] = fileparts(inputFile);
    outputFile = fullfile(outputFolder, [basename, '_output.mat']);

    % Check if the output file already exists
    if ~exist(outputFile, 'file')
        unprocessedFiles{end + 1} = inputFile;
    else
        disp(['Output file already exists, skipping: ', outputFile]);
    end
end

if ~isempty(unprocessedFiles)
    parfor i = 1:length(unprocessedFiles)
        inputFile = unprocessedFiles{i};
        [~, basename, ~] = fileparts(inputFile);
        outputFile = fullfile(outputFolder, [basename, '_output.mat']);

        try
            TS_Init(inputFile, 'hctsa', 0, outputFile);
            TS_Compute(false, [], [], 'missing', outputFile, true);

            disp(['HCTSA analysis completed for: ', inputFile]);
            disp(['Output file: ', outputFile]);
        catch ME
            disp(['Error processing file: ', inputFile]);
            disp(['Error message: ', ME.message]);
        end
    end
else
    disp('No unprocessed .mat files found in the specified folder.');
end