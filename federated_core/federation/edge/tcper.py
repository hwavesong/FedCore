# -*- coding: utf-8 -*-
import pickle
import socket
import struct


def sock_receive_content(sock, content_length):
    byted_content = bytes()
    while content_length >= 1024:
        recv_content = sock.recv(1024)
        byted_content += recv_content

        content_length -= len(recv_content)

    while content_length > 0:
        recv_content = sock.recv(content_length)
        byted_content += recv_content

        content_length -= len(recv_content)

    content = pickle.loads(byted_content)

    sock.close()

    return content


def weight_initialization_command(address):
    print('weigth initialization address', address)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(address)

        # w: weight
        # s: stop
        sock.send(struct.pack('ci', b'i', 200))


# HOST, PORT = "localhost", 9999
def weight_send(address, weights):
    byted_weigths = pickle.dumps(weights)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(address)

        # w: weight
        # s: stop
        sock.send(struct.pack('ci', b'w', len(byted_weigths)))

        sock.sendall(byted_weigths)


def weight_receive(address):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(address)
    server.listen()

    sock, addr = server.accept()

    # receive header information
    byted_content_header = sock.recv(struct.calcsize('ci'))
    byted_content_type, content_length = struct.unpack_from('ci', byted_content_header)
    content_type = byted_content_type.decode()

    if content_type != 'w':
        raise ValueError('error')

    # receive weight
    weight = sock_receive_content(sock, content_length)

    return weight


def weight_multi_receive(address, num_nodes):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(address)
    server.listen(num_nodes * 2)

    collected_weights = list()
    for idx_ in range(num_nodes):
        # print('receiving from {}'.format(idx_))

        sock, addr = server.accept()

        # receive header information
        byted_content_header = sock.recv(struct.calcsize('ci'))
        byted_content_type, content_length = struct.unpack_from('ci', byted_content_header)
        content_type = byted_content_type.decode()

        if content_type != 'w':
            raise ValueError('error')

        # receive weight
        weight = sock_receive_content(sock, content_length)

        collected_weights.append(weight)

    return collected_weights


# ===================================================
def evaluation_command(address):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(address)

        sock.send(struct.pack('ci', b'e', 200))


def evaluation_send(address, accuracy):
    # byted_accuracy = pickle.dumps(accuracy)
    #
    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    #     sock.connect(address)
    #
    #     sock.send(struct.pack('ci', b'w', len(byted_accuracy)))
    #
    #     sock.sendall(byted_accuracy)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(address)

        sock.send(struct.pack('cf', b'e', accuracy))


def evaluation_receive(address):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(address)
    server.listen()

    sock, addr = server.accept()

    byted_content_header = sock.recv(struct.calcsize('cf'))
    byted_content_type, accuracy = struct.unpack_from('cf', byted_content_header)
    content_type = byted_content_type.decode()

    if content_type != 'e':
        raise ValueError('error')

    return accuracy


# ===================================================
def shutdown_command(address):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(address)

        sock.send(struct.pack('ci', b's', 200))


# ===================================================

def received_events(address):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(address)
    server.listen()

    while True:
        sock, addr = server.accept()
        print('accept socket: {}'.format(addr))

        byted_content_header = sock.recv(struct.calcsize('ci'))
        byted_content_type, content_length = struct.unpack_from('ci', byted_content_header)
        content_type = byted_content_type.decode()

        if content_type == 'w':
            weight = sock_receive_content(sock, content_length)
            yield 'receive_weight', weight

        elif content_type == 'e':
            sock.close()
            yield 'evaluation', None
            # accuracy = sock_receive_content(sock, content_length)
            # yield 'evaluation', accuracy

        elif content_type == 'i':
            sock.close()
            yield 'initial_weight', None

        elif content_type == 's':
            sock.close()
            break

        else:
            yield 'error', None

    server.close()
