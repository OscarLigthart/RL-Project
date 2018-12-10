python CartPole.py --sampling off --target off

python CartPole.py --sampling uniform --target off

python CartPole.py --sampling prioritized --target off --memory_size 1000


python CartPole.py --sampling off --target soft

python CartPole.py --sampling uniform --target soft

python CartPole.py --sampling prioritized --target soft --memory_size 1000


python CartPole.py --sampling off --target hard

python CartPole.py --sampling uniform --target hard

python CartPole.py --sampling prioritized --target hard --memory_size 1000
