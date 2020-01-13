function unmixing_matrix = csp_geometric_approach_no_checks(data_path, output_path)
% Compute CSP in geometric approach without any checks.

    % load data
    % R_1: covariance matrix for the first class as (channels, channels)
    % R_2: covariance matrix for the second class as (channels, channels)
    load(data_path, 'R_1', 'R_2');
    
    % composite spatial covariance matrix
    R_c = R_1 + R_2;
    
    % factorize R_c into eigenvalues and eigenvectors
    [E, F] = eig(R_c);

    % eigenvalues are not ordered
    [F_sorted, ind] = sort(diag(F)); 
    % eigenvectors sorted in ascending order by corresponding eigenvalues
    E_sorted = E(:,ind); 
    
    % whitening transformation matrix
    U = diag(F_sorted.^(-0.5)) * E_sorted';
    
    % whiten spatial covariance matrix R_1
    S_1 = U * R_1 * U';
    
    % factorize S_1 into eigenvalues and eigenvectors
    [P, D_1] = eig(S_1);

    % eigenvalues are not ordered
    [D_1_sorted, ind] = sort(diag(D_1)); 
    % eigenvectors sorted in ascending order by corresponding eigenvalues
    P_sorted = P(:,ind); 
    
    % spatial filters are in rows
    W_T = P_sorted' * U;
    
    unmixing_matrix = W_T;
    eigenvalues = D_1_sorted;
        
    % save results
    save(output_path, 'unmixing_matrix', 'eigenvalues');
end
