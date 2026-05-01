from gridworld_env import GridWorldEnv


def main():
    env = GridWorldEnv(size=5, max_steps=20)
    state = env.reset()
    env.step(env.RIGHT)
    next_state, reward, done, info = env.step(env.DOWN)
    print(next_state, reward, done, info)

    # actions = [
    #     env.RIGHT,
    #     env.RIGHT,
    #     env.RIGHT,
    #     env.RIGHT,
    #     env.DOWN,
    #     env.DOWN,
    #     env.DOWN,
    #     env.DOWN,
    # ]

    # for action in actions:
    #     next_state, reward, done, info = env.step(action)
    #     print(
    #         "action:",
    #         action,
    #         "next_state:",
    #         next_state,
    #         "reward:",
    #         reward,
    #         "done:",
    #         done,
    #         "info:",
    #         info,
    #     )
    #     env.render()

    #     if done:
    #         break


if __name__ == "__main__":
    main()
