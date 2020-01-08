function unmixing_matrix = use_bbci(data_path, n_csp_components, output_path)
    setup_bbci();
    
    % load data
    load(data_path, 'trials', 'labels', 'fs', 'channel_names');
    [n_trials, n_channels, n_samples] = size(trials);

    % prepare data in BBCI format
    epo = {};
    epo.fs = fs; % sampling rate [samples per second]
    epo.x = permute(trials, [3 2 1]); % multichannel signals (DOUBLE [T #channels #epochs])

    for i = 1:n_channels
        epo.clab{i} = strtrim(channel_names(i, :)); % channel labels (CELL {1 #channels})
    end

    epo.y = zeros(2, n_trials); % class labels (DOUBLE [#classes #epochs])
    epo.y(1, labels == 0) = 1;
    epo.y(2, labels == 1) = 1;

    epo.className = {'0', '1'}; % class names (CELL {1 #classes})
    epo.t = (1:n_samples) / fs; % time axis (DOUBLE [1 T])

    % compute CSP
    % NOTE: it works only for even numbers of `n_csp_components`!
    [DAT, CSP_W, CSP_A, SCORE] = proc_csp(epo, 'SelectFcn', {@cspselect_equalPerClass, n_csp_components / 2});

    % CSP 'demixing' matrix (filters in columns)
    assert(all(size(CSP_W) == [n_channels, n_csp_components])) 
    % CSP 'mixing' matrix (patterns in rows)
    assert(all(size(CSP_A) == [n_channels, n_csp_components])) 

    % unmixing matrix
    unmixing_matrix = CSP_W';
    assert(all(size(unmixing_matrix) == [n_csp_components, n_channels]));
    
    % eigenvalues
    eigenvalues = SCORE;
    assert(length(eigenvalues) == n_csp_components);
        
    % save results
    save(output_path, 'unmixing_matrix', 'eigenvalues');
end

function setup_bbci()
    if ~exist('proc_csp', 'file')
        cd('/media/data/toolbox/bbci_public');
        startup_bbci_toolbox('DataDir', '/home/milan/bbci_data');
    end
end
