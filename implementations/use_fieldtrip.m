function unmixing_matrix = use_fieldtrip(data_path, n_csp_components, output_path)
    setup_fieldtrip();

    % load data
    load(data_path, 'trials', 'labels', 'fs', 'channel_names');
    [n_trials, n_channels, n_samples] = size(trials);

    % prepare data in FieldTrip format
    data = [];
    data.fsample = fs;

    for i = 1:n_channels
        data.label{i} = strtrim(channel_names(i, :)); % channel name
    end

    for i = 1:n_trials
        data.trial{i} = squeeze(trials(i, :, :)); % trial data
        data.time{i} = (1:n_samples) / fs; % trial time axis
    end

    data = ft_datatype_raw(data);  % only for checks

    % compute CSP
    cfg = [];
    cfg.method = 'csp';
    cfg.channel = 'all';
    cfg.trials = 'all';
    cfg.numcomponent = 'all';
    cfg.randomseed = 42;

    cfg.demean = 'no'; % optionally perform baseline correction on each trial 
    cfg.doscale = 'no'; % determine the scaling of the data, scale it to approximately unity
    cfg.updatesens = 'no'; % apply the linear projection also to the sensor description

    cfg.csp.classlabels = squeeze(labels + 1);  % class 1 or 2
    % NOTE: it works only for even numbers!
    cfg.csp.numfilters = n_csp_components;  % number of CSP components

    comp = ft_componentanalysis(cfg, data);

    % unmixing matrix
    unmixing_matrix = comp.unmixing;
    assert(all(size(unmixing_matrix) == [n_csp_components, n_channels]));

    % mixing matrix
    mixing_matrix = comp.topo;
    assert(all(size(mixing_matrix) == [n_channels, n_csp_components]));
    
    % save results
    save(output_path, 'unmixing_matrix');
end

function setup_fieldtrip()
    if ~exist('ft_defaults', 'file')
        addpath('/media/data/toolbox/fieldtrip')
        ft_defaults
    end
end
