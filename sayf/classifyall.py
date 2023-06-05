# import docker
# import random
import sys

# def build_image(docker_client, tag):
#     image_tag = "rl_assignment_" + str(tag)
#     path = "./"
#     docker_client.images.build(path=path, tag=image_tag)
#     return image_tag

 
if __name__ == '__main__':
    with open('readme.txt', 'w') as f:
        f.write('red\n')
        f.write('red\n')
        f.write('red\n')
        f.write('red\n')
        f.write('red\n')
        f.write('red\n')
        f.write('yellow\n')
        f.write('red\n')
    f.close()
    # docker_client = docker.from_env()
    # tag_number = 7307  # random.randint(0, 9999)
    # image_tag = build_image(docker_client, tag_number)

    # result_lines = docker_client.containers.run(image_tag, name=image_tag,
    #                                             stderr=True,
    #                                             remove=True,
    #                                             mem_limit="2GB")
    # result_lines = result_lines.splitlines()
    # if len(result_lines) > 0:
    #     print(float(result_lines[-1]))