folder = 'output_1000_device_2/'; 
files = dir(fullfile(folder, '*.csv')); 

disp(folder);

global output_folder;
output_folder = fullfile(folder, 'output_device_user_test');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Progress bar and timing setup
numFiles = length(files);
disp("Number of files:");
disp(numFiles);

startTime = tic;
progressBarLength = 50;

for i = 1:numFiles
    filename = fullfile(folder, files(i).name); % create the full file path
    T = readtable(filename, 'ReadVariableNames', false);
    secondRow = T{2, :};
    lastElement = secondRow{end};

    if strcmp(lastElement, 'Tilt Brush')
       continue;
    end
    
    outputFileNumber = ceil(i / 100);
    outputFileName = ['output_file', num2str(outputFileNumber), '.csv'];

    savedStart = 1;

    disp(['Processing file: ', filename, ' with startValue: ', num2str(savedStart)]);

    data = T; % Reuse the already read table
    num = table2array(data(:, 1:22));
    string_cells = table2cell(data(:, 23:end));

    timeValue = num(:, 1);
    endVal = (timeValue(end) - 10) * 10;

    Fs = 1 / mean(diff(timeValue));

    calculate = num;

    calculate(calculate(:, 1) < 10 | calculate(:, 1) > 80, :) = [];
    mainZ = calculate(:, 4);
    meanV = mean(mainZ);

    total_time = num(end, 1);
    total_loop = floor(total_time / 5);

    for x = 1:total_loop
        savedEnd = savedStart + 5;

        numPhone = table2array(data(:, 1:22));
        numPhone(numPhone(:, 1) < savedStart | numPhone(:, 1) > savedEnd, :) = [];

        if isempty(numPhone)
            disp('Data is empty here!');
        else
            my_xls(data, savedStart, savedEnd, endVal, string_cells, num, Fs, outputFileName);
        end
        
        savedStart = savedStart + 5;
    end
    
    progress = i / numFiles;
    elapsedTime = toc(startTime);
    remainingTime = elapsedTime * (1 - progress) / progress;

    numBars = round(progress * progressBarLength);
    progressBar = ['[' repmat('=', 1, numBars) repmat(' ', 1, progressBarLength - numBars) ']'];
    
    fprintf('Processing file %d of %d %s Estimated time remaining: %s\n', ...
            i, numFiles, progressBar, datestr(seconds(remainingTime), 'HH:MM:SS'));

    disp(['Finished processing file: ', filename]);
end

fprintf('All files processed. Total time elapsed: %s\n', datestr(seconds(toc(startTime)), 'HH:MM:SS'));

function my_xls(data, startValue, endValue, endVal, string_cells, num, Fs, outputFileName)
    class_str = string_cells{1, 3};
    class_str = string(class_str);

    numPhone = table2array(data(:, 1:22));
    comparePhone = numPhone;
    comparePhone(comparePhone(:, 1) < 10 | comparePhone(:, 1) > endVal + 10, :) = [];

    analysePhone = numPhone;
    numPhone(numPhone(:, 1) < startValue | numPhone(:, 1) > endValue, :) = [];

    totalStatX = [];

    for col = 2:22
        phoneZ = numPhone(:, col);
        comparePZ = comparePhone(:, col);

        Tr = linspace(numPhone(1, 1), numPhone(1, end), size(numPhone, 1));
        Dr = resample(phoneZ, Tr);
        Dr_mc  = Dr - mean(Dr, 1); 

        FDr_mc = fft(Dr_mc, [], 1);
        Fv = linspace(0, 1, fix(size(FDr_mc, 1) / 2) + 1) * Fs / 2;

        amplitude = abs(FDr_mc(1:numel(Fv), :)) * 2;
        upperPart = Fv * amplitude;
        ampSum = sum(amplitude);

        specCentroid = upperPart / ampSum;
        FvSqr = Fv .^ 2;
        stdDevupper = FvSqr * amplitude;
        specStdDev = sqrt(stdDevupper / ampSum);
        specCrest = max(amplitude) / specCentroid;

        specSkewness = (((Fv - specCentroid) .^ 3) * amplitude) / (specStdDev) ^ 3;
        specKurt = (sum((((amplitude - specCentroid) .^ 4) .* amplitude)) / (specStdDev) ^ 4) - 3;

        meanP = mean(phoneZ);
        meanPZ = mean(comparePZ);
        meanCrossingRateP = sum(phoneZ > meanPZ) / numel(phoneZ);

        statX = [mean(phoneZ), max(phoneZ), min(phoneZ), std(phoneZ), var(phoneZ), range(phoneZ), ...
                 (std(phoneZ) / mean(phoneZ)) * 100, skewness(phoneZ), kurtosis(phoneZ), ...
                 quantile(phoneZ, [0.25, 0.50, 0.75]), meanCrossingRateP, sum(pentropy(phoneZ, Fs)), ...
                 sum(amplitude) / max(amplitude), specCentroid, specStdDev, specCrest, ...
                 specSkewness, specKurt, max(Fv), max(phoneZ)];

        totalStatX = [totalStatX, statX];
    end

    totalStatX = [totalStatX, class_str];
    statX_cell = num2cell(totalStatX);

    global output_folder;
    outfilename = fullfile(output_folder, outputFileName);

    if exist(outfilename, 'file')
        writetable(cell2table(statX_cell), outfilename, 'WriteVariableNames', false, 'WriteMode', 'append');
    else
        writetable(cell2table(statX_cell), outfilename, 'WriteVariableNames', false);
    end
end
