# Trains unsupervised hmm

from hmm import unsupervised_HMM
import preprocess

def unsupervised_learning(n_states, N_iters):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''
    # genres - list of list of state seqences 
    # genre_map - maps word "obs" to numbers
    genres, genre_map = preprocess.process_shakespeare()

    # Train the HMM.
    HMM = unsupervised_HMM(genres, n_states, N_iters)

    # Print the transition matrix.
    print("Transition Matrix:")
    print('#' * 70)
    for i in range(len(HMM.A)):
        print(''.join("{:<12.3e}".format(HMM.A[i][j]) for j in range(len(HMM.A[i]))))
    print('')
    print('')

    # Print the observation matrix. 
    print("Observation Matrix:  ")
    print('#' * 70)
    for i in range(len(HMM.O)):
        print(''.join("{:<12.3e}".format(HMM.O[i][j]) for j in range(len(HMM.O[i]))))
    print('')
    print('')

if __name__ == '__main__':
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Code For Training"))
    print('#' * 70)
    print('')
    print('')

    unsupervised_learning(4, 1000)
