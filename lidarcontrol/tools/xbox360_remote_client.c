/* Read an Xbox 360 wireless receiver in userspace and forward controls via SSH. */
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <libusb-1.0/libusb.h>

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>

enum { VENDOR_ID = 0x045e, INTERFACE_CLASS = 0xff, INTERFACE_SUBCLASS = 0x5d,
       INTERFACE_PROTOCOL = 0x81 };
static const uint16_t product_ids[] = {0x0291, 0x0719, 0x02a9};
static volatile sig_atomic_t running = 1;

typedef struct {
    libusb_device_handle *handle;
    int interface_number;
    uint8_t endpoint_in;
    uint8_t endpoint_out;
} receiver_t;

static void on_signal(int signal_number) {
    (void)signal_number;
    running = 0;
}

static uint64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

static void delay_ms(long milliseconds) {
    struct timespec delay = {milliseconds / 1000, (milliseconds % 1000) * 1000000};
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR && running) {}
}

static bool supported_product(uint16_t product_id) {
    for (size_t i = 0; i < sizeof(product_ids) / sizeof(product_ids[0]); ++i) {
        if (product_ids[i] == product_id) return true;
    }
    return false;
}

static int find_receiver(libusb_context *context, receiver_t *receiver) {
    libusb_device **devices = NULL;
    ssize_t count = libusb_get_device_list(context, &devices);
    if (count < 0) return (int)count;
    int result = LIBUSB_ERROR_NO_DEVICE;
    for (ssize_t i = 0; i < count; ++i) {
        struct libusb_device_descriptor descriptor;
        if (libusb_get_device_descriptor(devices[i], &descriptor) != 0 ||
            descriptor.idVendor != VENDOR_ID || !supported_product(descriptor.idProduct)) {
            continue;
        }
        libusb_device_handle *handle = NULL;
        if (libusb_open(devices[i], &handle) != 0) continue;
        struct libusb_config_descriptor *config = NULL;
        if (libusb_get_active_config_descriptor(devices[i], &config) != 0 &&
            libusb_get_config_descriptor(devices[i], 0, &config) != 0) {
            libusb_close(handle);
            continue;
        }
        bool found = false;
        for (uint8_t iface = 0; iface < config->bNumInterfaces && !found; ++iface) {
            const struct libusb_interface *interface = &config->interface[iface];
            for (int alt = 0; alt < interface->num_altsetting && !found; ++alt) {
                const struct libusb_interface_descriptor *setting = &interface->altsetting[alt];
                if (setting->bInterfaceClass != INTERFACE_CLASS ||
                    setting->bInterfaceSubClass != INTERFACE_SUBCLASS ||
                    setting->bInterfaceProtocol != INTERFACE_PROTOCOL) continue;
                uint8_t in = 0, out = 0;
                for (uint8_t ep = 0; ep < setting->bNumEndpoints; ++ep) {
                    const struct libusb_endpoint_descriptor *endpoint = &setting->endpoint[ep];
                    if ((endpoint->bmAttributes & LIBUSB_TRANSFER_TYPE_MASK) !=
                        LIBUSB_TRANSFER_TYPE_INTERRUPT) continue;
                    if (endpoint->bEndpointAddress & LIBUSB_ENDPOINT_DIR_MASK)
                        in = endpoint->bEndpointAddress;
                    else
                        out = endpoint->bEndpointAddress;
                }
                if (in && out) {
                    receiver->handle = handle;
                    receiver->interface_number = setting->bInterfaceNumber;
                    receiver->endpoint_in = in;
                    receiver->endpoint_out = out;
                    found = true;
                }
            }
        }
        libusb_free_config_descriptor(config);
        if (found) {
            result = 0;
            break;
        }
        libusb_close(handle);
    }
    libusb_free_device_list(devices, 1);
    return result;
}

static int send_json(int fd, const char *message) {
    size_t length = strlen(message);
    size_t sent = 0;
    while (sent < length) {
        ssize_t result = send(fd, message + sent, length - sent, 0);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) return -1;
        sent += (size_t)result;
    }
    return 0;
}

static int connect_control(const char *port_text) {
    char *end = NULL;
    long port = strtol(port_text, &end, 10);
    if (!end || *end || port < 1 || port > 65535) {
        fprintf(stderr, "Port de contrôle invalide : %s\n", port_text);
        return -1;
    }
    struct sockaddr_in address = {0};
    address.sin_family = AF_INET;
    address.sin_port = htons((uint16_t)port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    uint64_t deadline = now_ms() + 180000;
    while (running && now_ms() < deadline) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0) {
            int no_sigpipe = 1;
            setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
            if (connect(fd, (struct sockaddr *)&address, sizeof(address)) == 0) return fd;
            close(fd);
        }
        delay_ms(250);
    }
    fprintf(stderr, "Tunnel de commande SSH indisponible après 180 secondes.\n");
    return -1;
}

static int16_t read_i16_le(const uint8_t *bytes) {
    return (int16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage : xbox360_remote_client <port-local-du-tunnel>\n");
        return 2;
    }
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    libusb_context *context = NULL;
    int rc = libusb_init(&context);
    if (rc != 0) {
        fprintf(stderr, "Initialisation libusb impossible : %s\n", libusb_error_name(rc));
        return 1;
    }
    receiver_t receiver = {0};
    rc = find_receiver(context, &receiver);
    if (rc != 0) {
        fprintf(stderr, "Récepteur Xbox 360 Microsoft compatible introuvable (045e:0291/0719/02a9).\n");
        libusb_exit(context);
        return 1;
    }
    (void)libusb_set_auto_detach_kernel_driver(receiver.handle, 1);
    rc = libusb_claim_interface(receiver.handle, receiver.interface_number);
    if (rc != 0) {
        fprintf(stderr, "Impossible d'ouvrir l'interface USB du récepteur : %s\n", libusb_error_name(rc));
        libusb_close(receiver.handle);
        libusb_exit(context);
        return 1;
    }
    const uint8_t presence_query[12] = {0x08, 0x00, 0x0f, 0xc0};
    int transferred = 0;
    (void)libusb_interrupt_transfer(receiver.handle, receiver.endpoint_out,
                                    (unsigned char *)presence_query, sizeof(presence_query),
                                    &transferred, 1000);
    printf("Récepteur Xbox 360 détecté. Appuie sur Connect si la manette n'est pas appairée.\n");
    fflush(stdout);

    int control_fd = connect_control(argv[1]);
    if (control_fd < 0) {
        libusb_release_interface(receiver.handle, receiver.interface_number);
        libusb_close(receiver.handle);
        libusb_exit(context);
        return 1;
    }
    printf("Tunnel SSH connecté. RT/LT conduisent, le joystick gauche dirige, A enregistre, LB arrête.\n");
    fflush(stdout);

    uint8_t packet[64];
    uint64_t last_sent = 0;
    uint8_t last_buttons = 0xff;
    bool pad_present = false;
    while (running) {
        transferred = 0;
        rc = libusb_interrupt_transfer(receiver.handle, receiver.endpoint_in, packet,
                                       sizeof(packet), &transferred, 250);
        if (rc == LIBUSB_ERROR_TIMEOUT) continue;
        if (rc != 0) {
            fprintf(stderr, "Lecture USB arrêtée : %s\n", libusb_error_name(rc));
            break;
        }
        if (transferred >= 2 && (packet[0] & 0x08)) {
            bool present = (packet[1] & 0x80) != 0;
            if (present != pad_present) {
                printf(present ? "Manette Xbox 360 connectée.\n" : "Manette déconnectée : moteur arrêté par délai de sécurité.\n");
                fflush(stdout);
                pad_present = present;
            }
        }
        if (transferred < 18 || packet[1] != 0x01 || packet[4] != 0x00) continue;
        const uint8_t *report = packet + 4;
        uint8_t button_byte = report[3];
        bool a = (button_byte & 0x10) != 0;
        bool lb = (button_byte & 0x01) != 0;
        uint64_t current = now_ms();
        if (current - last_sent < 50 && button_byte == last_buttons && !lb) continue;
        char message[160];
        int length = snprintf(message, sizeof(message),
                              "{\"rt\":%u,\"lt\":%u,\"lx\":%d,\"a\":%s,\"lb\":%s}\n",
                              report[5], report[4], read_i16_le(report + 6),
                              a ? "true" : "false", lb ? "true" : "false");
        if (length <= 0 || (size_t)length >= sizeof(message) ||
            send_json(control_fd, message) != 0) {
            fprintf(stderr, "Connexion de contrôle perdue.\n");
            break;
        }
        last_sent = current;
        last_buttons = button_byte;
        if (lb) break;
    }
    (void)send_json(control_fd, "{\"type\":\"stop\"}\n");
    close(control_fd);
    libusb_release_interface(receiver.handle, receiver.interface_number);
    libusb_close(receiver.handle);
    libusb_exit(context);
    return 0;
}
