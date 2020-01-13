function unmixing_matrix = csp_gep_no_checks(data_path, output_path)
% Compute CSP as generalized eigenvalue problem without any checks.

    % load data
    % R_1: covariance matrix for the first class as (channels, channels)
    % R_2: covariance matrix for the second class as (channels, channels)
    load(data_path, 'R_1', 'R_2');
    
    % generalized eigenvalue problem
    [W, D_1] = eig(R_1, R_1 + R_2);
    
    % eigenvalues are not ordered
    [D_1_sorted, ind] = sort(diag(D_1)); 
    % eigenvectors sorted in ascending order by corresponding eigenvalues
    W_sorted = W(:,ind); 

    % transpose as eigenvectors (spatial filters) are in columns of W
    unmixing_matrix = W_sorted';
    eigenvalues = D_1_sorted;
        
    % save results
    save(output_path, 'unmixing_matrix', 'eigenvalues');
end
