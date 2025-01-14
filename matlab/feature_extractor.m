folder = '/media/mahdad/easystore/utility_codes/media/Downloads/output_100/'; % specify the folder where your files are
files = dir(fullfile(folder, '*.csv')); % get all .csv files in the folder



global output_folder;
output_folder = fullfile(folder, 'output');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%progress bar code

% Get the total number of files
numFiles = length(files);



% Store the start time
startTime = tic;

% Define the length of the progress bar
progressBarLength = 50;

for i = 1:length(files)
    
     

    filename = fullfile(folder, files(i).name); % create the full file path
    
    % Determine the output file number based on the current index
    outputFileNumber = ceil(i / 100);
    
   % Create the output file name
    outputFileName = ['output_file', num2str(outputFileNumber), '.csv'];
    
    disp(outputFileName);
    
    
   
    fnameString="test"; 
    fnameNeumericStart=1; 

    
    savedStart = 1;
    class_str2="robo";
   
    
    % Display the filename and startValue
    disp(['Processing file: ', filename, ' with startValue: ', num2str(savedStart)]);
    %savedEnd=savedStart+10;
    
    % Read the CSV file into a table
    data = readtable(filename);

    % Extract the first 22 columns as a numeric matrix
    num = table2array(data(:, 1:22));

    % Extract the remaining columns as a cell array of strings
    string_cells = table2cell(data(:, 23:end));
    
    %disp(string_cells);
     % Extract the remaining columns as a cell array of strings
    % Extract the 23rd column as a cell array of strings
        %string_cells = table2cell(data(:, 23));


    %num = csvread(filename) ;
    [r,c] = size(num) ;
    class_str = string_cells{1, 2};
    %disp(class_str);  % Display the value of class_str for debugging purposes
    class_str = string(class_str);  % Convert it to a string
    %disp(class_str);  % Display the converted value of class_str
    %disp(clsss(class_str));
   % testClass="robo";
    %disp(class(testClass));
    
    timeValue=num(:,1); 
    endVal=timeValue(end);
    endVal=(endVal-10)*10;
    %disp(endVal);

    %Delete unecessary information
    %num(23:end,:) = [];
    

    regionCount=0;
    consecutiveChecker=0;
    consecutiveStart=0;
    consecutiveEnd=0;


    zerocount=0;
    onecount=0;
    twocount=0;
    threecount=0;
    fourcount=0;
    fivecount=0;
    sixcount=0;
    sevencount=0;
    eightcount=0;
    ninecount=0;

    %Fs = 1/mean(diff(num(:,1)));  
    %y_highpass=highpass(num(:,4),20,Fs);
    %num(:,4)=y_highpass;
    
    %high pass filter
    Fs = 1/mean(diff(num(:,1)));  
    %y_highpass=highpass(num(:,4),30,Fs);
    %num(:,4)=y_highpass;


    calculate=num;

     %Delete rows for specific condition
      clowIndices = find(calculate(:,1)<10);
      calculate(clowIndices,:) = []; 

      chighIndices = find(calculate(:,1)>80);
      calculate(chighIndices,:) = [];

      mainZ=calculate(:,4) ;
      meanV=mean(mainZ);

      %disp(mainZ)



    notInside=0;
    startObserver=0;
    startConsecutive=0;
    appStart=0;
    endConsecutive=0;
    tempEnd=0;
    appEnd=0;
    endObserver=0;
    regionCount=0;
    errorCount=0;
    errorState=0;  
    largeCount=0;
    smallCount=0;
    %xlimit=10;
    
   
    total_time = num(end, 1);
    total_loop=total_time/5;
    
    %disp("showing Total loop")
    %disp(total_loop)
    
    %disp("showing total time");
    %disp(total_time);


     for x=1:total_loop
        %disp(savedStart);

        savedEnd=savedStart+5;
        %disp(x);
        my_xls(data,savedStart,savedEnd,endVal,string_cells,num,Fs,outputFileName);
        savedStart=savedStart+5;
     end
    
     % Calculate progress
    progress = i / numFiles;
    
    % Calculate and display estimated time remaining
    elapsedTime = toc(startTime);
    remainingTime = elapsedTime * (1 - progress) / progress;
    
    % Generate the progress bar string
    numBars = round(progress * progressBarLength);
    progressBar = ['[' repmat('=', 1, numBars) repmat(' ', 1, progressBarLength - numBars) ']'];
    
    % Print the progress bar and estimated time remaining
    fprintf('Processing file %d of %d %s Estimated time remaining: %s\n', ...
            i, numFiles, progressBar, datestr(seconds(remainingTime), 'HH:MM:SS'));
    
    % Optional: Pause for a short period to simulate processing time (for demonstration purposes)
    % pause(0.1);
    
     % Display finished processing message
    disp(['Finished processing file: ', filename]);
end

% Final message
%fprintf('All files processed. Total time elapsed: %s\n', datestr(seconds(elapsedTime), 'HH:MM:SS'));

function my_xls(data,start,funcend,endVal,string_cells,num,Fs,outputFileName)



startValue=start;
endValue=funcend;
%watchCompareStart=20;
%watchCompareEnd=34;
phoneCompareStart=10;
phoneCompareEnd=endVal+10;
%disp(watchStartValue);
%disp(endValue);
%disp(watchEndValue);

class_str = string_cells{1, 2};
%disp(class_str);  % Display the value of class_str for debugging purposes
class = string(class_str);  % Convert it to a string

%now starting the smartphonw calculations

 % Read the CSV file into a table
    %data = readtable(filename);

    % Extract the first 22 columns as a numeric matrix
    numPhone = table2array(data(:, 1:22));
   


[r1,c1] = size(numPhone) ;

%numPhone(2:2:end,:) = [] ;
%numPhone(23:end,:) = [];



comparePhone=numPhone;

%Delete rows for phone compare values
comparePLow = find(comparePhone(:,1)<phoneCompareStart);
comparePhone(comparePLow,:) = [];

comparePHigh = find(comparePhone(:,1)>phoneCompareEnd);
comparePhone(comparePHigh,:) = [];

FsT= 1/mean(diff(numPhone(:,1)));  



%Delete rows for specific condition in Phone
lowIndicesp = find(numPhone(:,1)<startValue);
numPhone(lowIndicesp,:) = [];

highIndicesp = find(numPhone(:,1)>endValue);
numPhone(highIndicesp,:) = [];


analysePhone=numPhone;

Fsp = 1/mean(diff(numPhone(:,1)));  
Fn=Fsp/2;
%y_highpass=highpass(numPhone(:,4),30,Fsp);
%numPhone(:,4)=y_highpass;



phoneRms = numPhone(:,5) ;

totalStatX = [];

for col = 2:22
    phoneZ=numPhone(:,col);
    comparePZ=comparePhone(:,col);

    %silent time code

    %[silentTime,totalPeak]=silentTimeSelector(analysePhone,startValue);

%fprintf('Total spiked percentage %d\n',totalPeak);  
%fprintf('Total Silent Time %.2f\n', silentTime);

%Calculating Frequency domain features


    Tr = linspace(numPhone(1,1), numPhone(1,end), size(numPhone,1));  
    Dr = resample(phoneZ, Tr); 
    Dr_mc  = Dr - mean(Dr,1); 


    FDr_mc = fft(Dr_mc, [], 1);
    Fv = linspace(0, 1, fix(size(FDr_mc,1)/2)+1)*Fn; 

    Iv = 1:numel(Fv); 
    amplitude=abs(FDr_mc(Iv,:))*2;

    upperPart=Fv*amplitude;
    ampSum=sum(amplitude);

    specCentroid=upperPart/ampSum;
    %disp(specCentroid); 

    FvSqr=Fv.^2;
    stdDevupper=FvSqr*amplitude;
    specStdDev=sqrt(stdDevupper/ampSum);
    specCrest=max(amplitude)/specCentroid;


    specSkewness=(((Fv-specCentroid).^3)*amplitude)/(specStdDev)^3;

    specKurt=(sum((((amplitude-specCentroid).^4).*amplitude))/(specStdDev)^4)-3 ;
    maxFreq=max(Fv);
    maxMagx=max(phoneZ);
    







    meanP=mean(phoneZ);
    minP=min(phoneZ);
    maxP=max(phoneZ);
    meanPZ=mean(comparePZ);
    %gradientZ=mean(gradient(phoneZ));
    %disp(meanPZ); 
    irrk=irregularityk(phoneZ);
    irrj=irregularityj(phoneZ);
    sharp=sharpness(phoneZ);
    smooth=smoothness(phoneZ);

%now adding frequency domain things:





%disp(meanP);
%disp(minP);
%disp(maxP);

    meanCrossingP=phoneZ > meanPZ;
    numberCrossingP=sum(meanCrossingP(:) == 1);
    meanCrossingRateP=numberCrossingP/numel(phoneZ);
%disp(meanCrossingRateP);



%Extracting frequency domain values:

    Fp = fft(phoneZ,1024);
    FFTCoEffp=Fp/length(phoneZ);
    powp = Fp.*conj(Fp);
    total_powp = sum(powp);
%disp(total_powp);



    Fsp = 1/mean(diff(numPhone(:,1)));

    penp=pentropy(phoneZ,Fsp);
    sumPenp=sum(penp);




    hdp = dfilt.fftfir(phoneZ,1024);
    cp=fftcoeffs(hdp);
    ampp = 2*abs(cp)/length(phoneZ);
    phasep=angle(cp);
    magnitudep=abs(ampp);

    highestMagp=max(magnitudep);
    sumMagp=sum(magnitudep);

    frequency_ratiop=highestMagp/sumMagp;


%now the signal shape features

% Number of peaks
    numPeaks = length(findpeaks(phoneZ));

% Zero-crossing rate
    zeroCrossingRate = length(find(phoneZ(1:end-1).*phoneZ(2:end) < 0));

% Slope changes
    slopeChanges = length(find(diff(phoneZ(2:end)) .* diff(phoneZ(1:end-1)) < 0));

    % Number of inflection points
    numInflectionPoints = length(find(diff(diff(phoneZ)) > 0));

   

    %statText=[mean(phoneZ) max(phoneZ) min(phoneZ) std(phoneZ) var(phoneZ) range(phoneZ) (std(phoneZ)/mean(phoneZ))*100 skewness(phoneZ) kurtosis(phoneZ) quantile(phoneZ,[0.25,0.50,0.75]) meanCrossingRateP total_powp sumPenp frequency_ratiop irrk irrj sharp smooth specCentroid specStdDev specCrest specSkewness specKurt maxFreq maxMagx numPeaks zeroCrossingRate slopeChanges numInflectionPoints;
    statX=[mean(phoneZ) max(phoneZ) min(phoneZ) std(phoneZ) var(phoneZ) range(phoneZ) (std(phoneZ)/mean(phoneZ))*100 skewness(phoneZ) kurtosis(phoneZ) quantile(phoneZ,[0.25,0.50,0.75]) meanCrossingRateP total_powp sumPenp frequency_ratiop irrk irrj sharp smooth specCentroid specStdDev specCrest specSkewness specKurt maxFreq maxMagx numPeaks zeroCrossingRate slopeChanges numInflectionPoints];
    %disp(class(class_str));
    %disp(class(class_str2));
    %disp(statX);
    totalStatX = [totalStatX, statX];
    
end


%disp(mean(phoneZ));
t=[class];

 totalStatX = [totalStatX, t];








%disp(nextRow);
%global nextRow;
%nextRow=nextRow+1;

% Write the outputValues to the appropriate columns in the Excel file
%cellReference = sprintf('A%d', nextRow);
%xlswrite('acctest.xls', statX, 'Sheet1', cellReference);
% Convert statX to a row matrix

% Convert statX to a row cell array
statX_cell = num2cell(totalStatX);

% Specify the file name
global output_folder;
outfilename =fullfile(output_folder, outputFileName);

% Append statX to the CSV file
if exist(outfilename, 'file')
    % If the file already exists, append to it
    writetable(cell2table(statX_cell), outfilename, 'WriteVariableNames', false, 'WriteMode', 'append');
else
    % If the file doesn't exist, create a new one
    writetable(cell2table(statX_cell), outfilename, 'WriteVariableNames', false);
end


end


function ikSum=irregularityk(phonez)
%disp(phonez);
N=[10 100 1000];

ikSum=0;
for i=1:length(phonez)-2
   ik=(phonez(i+1)-(phonez(i)+phonez(i+1)+phonez(i+2)/3));
   ikSum=ikSum+ik;
    
end
end

function ijSum=irregularityj(phonez)


ijSum=0;
for i=1:length(phonez)-1
   ij1=(phonez(i)-phonez(i+1))^2;
   ij2=phonez(i)^2;
   ij=ij1/ij2;
   ijSum=ijSum+ij;
    
end
end

function finalsharp=sharpness(phonez)
sharpn=0;
tempi=0;
for i=1:length(phonez)
    if(i<15)
        tempi=real(i*phonez(i)^0.23);
        %disp(tempi);
    else
        tempi=real(0.066*exp(0.171*i)*i*phonez(i)^0.23);
    end
    
    sharpn=sharpn+tempi;
end

finalsharp=(0.11*sharpn)/length(phonez);
end

function smoothSum=smoothness(phonez)

smoothSum=0;
for i=1:length(phonez)-2
   
    ismooth=real((20*log(phonez(i))-(20*log(phonez(i))+20*log(phonez(i+1))+20*log(phonez(i+2))))/3);
    
    smoothSum=smoothSum+ismooth;
end


end








