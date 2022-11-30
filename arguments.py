import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Autoencoders for data reconstruction",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-bs', '--batch', type=int, default=16,
                        help='Number of samples that will be propagated through the network')

    parser.add_argument('-e', '--epochs', type=int, default=500,
                        help='max num of epochs')

    parser.add_argument('-pt', '--patience', type=int, default=5,
                        help='patienece')

    parser.add_argument('-pa', '--path', type=str, default='IAQ_2month_Vah.mat',
                        help='data path')

    parser.add_argument('-md', '--memdim', type=int, default=149,
                        help='memory dimension')

    parser.add_argument('-mo', '--model', type=str, default='AE',
                        choices=[
                            'AE', 'DAE', 'MAE', 'VAE', 'MVAE'
                        ],
                        help=' model selection')

    parser.add_argument('-sf', '--failure', type=str, default='Complete_failure',
                        choices=[
                            'Bias', 'Complete_failure','Drift', 'Degradation'
                        ],
                        help=' failure type')

    parser.add_argument('-psta', '--pstart', type=int, default=0,
                        help='plot start point')

    parser.add_argument('-psto', '--pstop', type=int, default=700,
                        help='plot stop point')

    parser.add_argument('-fsta', '--fstart', type=int, default=400,
                        help='Fault period start point')

    parser.add_argument('-fsto', '--fstop', type=int, default=800,
                        help='Fault period stop point')

    parser.add_argument('-fm', '--fmagnitude', type=int, default=2,
                        help='Fault magnitude')

    parser.add_argument('-fs', '--fsensor', type=str, default='PM25',
                        choices=[
                            'PM25', 'PM10', 'CO2', 'Temp','Humidity'
                        ],
                        help=' Faulty sensor')

    return parser.parse_args()
