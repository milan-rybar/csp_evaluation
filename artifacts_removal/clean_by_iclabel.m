% Clean data using ICLabel package from EEGLABEL.
% Beware of hard-coded paths!
clear; clc;  % Clear memory and the command window

filename = 'data_set_IVa_aa';

% move to EEGLAB directory
cd('/media/data/toolbox/eeglab');

% load data
load(['/home/milan/csp_evaluation/data/' filename '.mat']);

% EEGLAB: channels, time
eeg_data = double(cnt') * 0.1; % convert to uV values
n_channels = size(eeg_data, 1);
n_samples = size(eeg_data, 2);

% load dataset in EEGLAB
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_importdata('dataformat','array','nbchan',0,'data','eeg_data','setname','dataset','srate',nfo.fs,'pnts',0,'xmin',0);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'gui','off'); 
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);

% filter data
EEG = pop_eegfiltnew(EEG, 'locutoff',1,'hicutoff',40);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'gui','off'); 

% add channel locations
EEG = eeg_checkset( EEG );
EEG = pop_chanedit(EEG, 'load',{'/home/milan/csp_evaluation/data/locations.xyz' 'filetype' 'autodetect'});
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

% train ICA
EEG = eeg_checkset( EEG );
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

% run IC Label
EEG = iclabel(EEG);

% get artifact components
[m, idx] = max(EEG.etc.ic_classification.ICLabel.classifications, [], 2);
ic_artifacts = [];
ic_classes = [];
for i = 1:length(m)
    if idx(i) ~= 1 && idx(i) ~= 7 && m(i) > 1/7
        ic_artifacts = [ic_artifacts i];
        ic_classes = [ic_classes idx(i)];
    end
end

ic_classes_name = EEG.etc.ic_classification.ICLabel.classes(ic_classes);

% remove components
EEG = eeg_checkset( EEG );
EEG = pop_subcomp( EEG, ic_artifacts, 0);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 3,'gui','off'); 

% save EEGLAB dataset
EEG = eeg_checkset( EEG );
EEG = pop_saveset( EEG, 'filename',[filename '.set'],'filepath','/home/milan/csp_evaluation/results/iclabel');
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

% save results
cnt = EEG.data';
cnt = cnt / 0.1; % scale to original range
save(['/home/milan/csp_evaluation/results/iclabel/' filename '.mat'], 'mrk', 'nfo', 'cnt', 'ic_artifacts', 'ic_classes', 'ic_classes_name');
