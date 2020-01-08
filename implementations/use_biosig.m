function unmixing_matrix = use_biosig(data_path, n_csp_components, output_path)
    setup_biosig();
    
    % load data
    load(data_path, 'trials', 'labels');
    [n_trials, n_channels, n_samples] = size(trials);

    X = trials(labels == 0, :, :);
    Y = trials(labels == 1, :, :);

    % concatenate all trials into shape (trials * time, channels)
    X = reshape(X, n_channels, size(X, 1) * n_samples);
    X = permute(X, [2 1]);
    Y = reshape(Y, n_channels, size(Y, 1) * n_samples);
    Y = permute(Y, [2 1]);
    
    % computes only 4 spatial filters (2 per each class)
    [V, D] = csp(X, Y); % uses covm(X, 'E') inside

    assert(all(size(V) == [n_channels, 4])) % 4 spatial filters
    assert(all(size(D) == [n_channels, n_channels])) % eigenvalues in trace
    
    % unmixing matrix
    unmixing_matrix = V';
    assert(all(size(unmixing_matrix) == [4, n_channels]));
    
    % save results
    save(output_path, 'unmixing_matrix');
end

function setup_biosig()
    if ~exist('csp', 'file')
        addpath(genpath('/media/data/toolbox/biosig-code/biosig4matlab'))
    end
end
