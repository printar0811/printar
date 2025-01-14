% Simplified version to test file reading
folder = 'output_10000/'; 
files = dir(fullfile(folder, '*.csv')); 
disp(folder);

for i = 1:length(files)
    filename = fullfile(folder, files(i).name);
    disp(['Processing file: ', filename]);
    
    % Test reading the table
    try
        T = readtable(filename, 'ReadVariableNames', false);
        disp(T(1:5, :)); % Display first few rows
    catch ME
        disp(['Error reading file: ', filename]);
        disp(getReport(ME));
    end
end
