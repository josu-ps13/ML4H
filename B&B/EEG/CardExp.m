%% ML4H - Biosignals & Bioimages
% Practice 3: EEG - Card Experiment

% 100384150 · María González García
% 100405565 · Marta González García
% 100409398 · Josué Pérez Sabater

close all; clearvars, clc

w       = 100; % width to extend stimulus interval (see stimulus_detection function)
fc_low  = 1.5; % low cutoff frequency (Hz)
fc_high = 25;  % high cutoff frequency (Hz)
wdw_bef = 0.1; % window size before stimulus onset (s)
wdw_aft = 0.7; % window size after stimulus onset (s)

load('CardExperiment_23.mat') % Load file containing signals
%  - fs       sampling frequency (250 Hz)
%  - thrJ     threshold value for Joker card (4300000)
%  - thrQ     threshold value for Queen card (3800000)
%  - thrK     threshold value for King card (3400000)
%  - eeg      EEG signal
%  - trig     trigger signal (which card is presented)

t = 0:1/fs:(length(trig)-1)/fs; % time (for both EEG and trigger signal)
eeg = eeg';                     % convert column vectors to row vectors
trig = trig';
bef = round(wdw_bef*fs);        % convert seconds to samples
aft = round(wdw_aft*fs);

%% Find the position at the onset of the stimulus

% Apply median filter to trigger signal
trig_filt = medfilt1(trig, 'truncate');

[trig_temp, ini_J] = stimulus_detection(trig_filt, thrJ, w, bef, aft);
[trig_temp, ini_Q] = stimulus_detection(trig_temp, thrQ, w, bef, aft);
[        ~, ini_K] = stimulus_detection(trig_temp, thrK, w, bef, aft);

% Plot trigger signal, together with detected stimulus
figure, subplot(311), hold on
plot(t, trig_filt, color=[.3 .3 .3 .5])
scatter(ini_J./fs, repmat(thrJ, size(ini_J)), 'xg')
scatter(ini_Q./fs, repmat(thrQ, size(ini_Q)), 'xr')
scatter(ini_K./fs, repmat(thrK, size(ini_K)), 'xk')
title('Detected triggers')
ylabel('Amplitude (AU)')
legend('Trigger signal', 'Joker', 'Queen', 'King')
xlim([10 30])

%% Filter the EEG channel

% Design and apply band-pass filter to the EEG signal
[b, a] = butter(3, [fc_low/(fs/2), fc_high/(fs/2)]);
eeg_filt = filtfilt(b, a, eeg);
eeg_filt = eeg_filt - mean(eeg_filt); % set filtered EEG signal mean to 0

% Plot filtered EEG signal, together with detected stimulus
subplot(312), hold on
plot(t, eeg_filt, color=[.3 .3 .3 .3])
scatter(ini_J./fs, zeros(size(ini_J)), 'xg')
scatter(ini_Q./fs, zeros(size(ini_Q)), 'xr')
scatter(ini_K./fs, zeros(size(ini_K)), 'xk')
title('EEG signal filtered (band-pass)')
ylabel('Potential')
legend('Filtered EEG signal', 'Joker', 'Queen', 'King')
xlim([10 30])

%% Compute P300 wave for each card

J = P300(eeg_filt, ini_J, bef, aft);
Q = P300(eeg_filt, ini_Q, bef, aft);
K = P300(eeg_filt, ini_K, bef, aft);

timeJ = (1:length(J))/fs;
timeQ = (1:length(Q))/fs;
timeK = (1:length(K))/fs;

% Plot the three P300 waves
subplot(313), hold on
plot(timeJ, J, 'g')
plot(timeQ, Q, 'r')
plot(timeK, K, 'k')
xline(wdw_bef, '--')
title('P300 for each card')
xlabel('Time (s)');
ylabel('Potential');
legend('Joker', 'Queen', 'King')
xlim([0 wdw_bef+wdw_aft])

set(findobj(type='ax'), FontSize=15)      % increase font size
set(findobj(type='line'), LineWidth=2)    % increase line width
set(findobj(type='scatter'), LineWidth=2) % increase line width
set(findobj(type='fig'), color='w')       % make background of figures white

%% FUNCTIONS

function[trig, ini_st] = stimulus_detection(trig, thr, w, bef, aft) 
    % This function receives a trigger signal and detects the positions at
    % which the signal exceeds a threshold value, which correspond to the
    % beginning of a stimulus. In order to sequentially detect lower
    % thresholds, we set to zero the already-detected stimuli and return
    % the new trigger signal with the detected stimuli removed (together
    % with the initial positions of each stimulus).

    % The size of the stimulus is extended before and after by a size of w
    % to make sure we remove the full stimulus. Because of that, trigger
    % signal needs to be zero-padded by the same size.

    %  - trig    trigger signal
    %  - thr     threshold to detect stimulus
    %  - w       width to extend stimulus interval
    %  - bef     window size before stimulus onset
    %  - aft     window size after stimulus onset
    %  - stim    vector determining at each position whether there is stimulus
    %  - ini_st  initial position (onset) of each stimulus
    %  - end_st  ending position of each stimulus

    trig_pad = [zeros(1,w) trig zeros(1,w)]; % zero padding
    stim = trig_pad>thr;                     % index of detected stimulus
    ini_st = find(diff(stim)==1);            % beginning of stimulus
    end_st = find(diff(stim)==-1);           % end of stimulus

    % Each stimulus window is extended before and after by a size of w.
    for i = 1:length(ini_st)
        stim(ini_st(i)-w+1: ini_st(i)) = 1;
        stim(end_st(i):end_st(i)+w-1) = 1;
    end

    % Remove zero padding.
    stim = stim(1+w:end-w);
    ini_st = ini_st - w;

    % If a stimulus lies too close to the beginning or end of the signal,
    % its window may exceed time axis so we choose to remove it. We compare
    % with the size of the window before and after the onset.
    if ini_st(1) < bef, ini_st(1) = []; end
    if length(trig) - ini_st(end) < aft, ini_st(end) = []; end

    % Values at positions where a stimulus was detected are removed from
    % original trigger signal.
    trig = ~stim .* trig;
end

function[p300] = P300(eeg, ini, bef, aft)
    % This function receives an EEG signal together with the positions that
    % correspond to the beginning (onset) of all stimuli for a given card.
    % The function returns the P300 waveform, which is the average EEG
    % signal of all windows where a stimulus occurs.

    % It is common practise to include 0.1 seconds before the onset of the
    % stimulus, compute the mean of this interval and substract it to the
    % part of the signal after the onset, in order to alleviate any mean
    % difference before the stimulus.

    % Stimulus window extends from a few ms before the onset (for the
    % reason explained above) to a few ms after the onset. These intervals
    % are defined by variables bef and aft, respectively.

    %  - eeg   EEG signal
    %  - ini   initial position of the stimulus, corresponding to its onset
    %  - bef   stimulus window size, taken previous to the onset
    %  - aft   stimulus window size, taken after the onset
    %  - p300  P300 wave, averaged for all windows

    p300 = zeros(length(ini), bef+aft+1);

    for i = 1:length(ini)
        wdw = eeg(ini(i)-bef : ini(i)+aft); % EEG during whole stimulus window
        wdw_bef= wdw(1:bef);                % EEG before onset
        p300(i,:) = wdw - mean(wdw_bef);    % subtract mean of EEG before onset
    end
    
    p300 = mean(p300); % make the average of all windows
end
