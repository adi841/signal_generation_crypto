
import zmq
from zmq.utils.monitor import parse_monitor_message
import functools
import inspect

from .dataclass_utils import SocketStatus

@functools.lru_cache(maxsize=2)
def get_events_dict():
    event_key_dict = {value: key for key, value in zmq.__dict__.items() if key.startswith('EVENT_')}
    return event_key_dict

##
async def async_event_monitoring(monitor_socket, configObject, callback=None):
        
    try:        
        msg = await monitor_socket.recv_multipart()
        event = parse_monitor_message(msg)

        event_dict = get_events_dict()
        event_name = event_dict.get(event['event'], "UNKNOWN")
        msg_ = f"{event_name}. Value is: {event['value']}, Address is: {event['endpoint']}"
        configObject.masterlog.get_logger("zmq_monitor", "info")(msg_)
        
        if callback is not None:
            if inspect.iscoroutinefunction(callback):
                await callback(event)
    
            else:
                callback(event)

    except Exception as e:
        import traceback
        tb_ = traceback.format_exc()
        msg_ = f"Exception in event_monitoring: {e} \n {tb_}"
        configObject.masterlog.get_logger("zmq_monitor", "error")(msg_)

def sync_event_monitoring(monitor_socket, configObject, callback=None):
        
    try:        
        msg = monitor_socket.recv_multipart()
        event = parse_monitor_message(msg)

        event_dict = get_events_dict()
        event_name = event_dict.get(event['event'], "UNKNOWN")
        msg_ = f"{event_name}. Value is: {event['value']}, Address is: {event['endpoint']}"
        configObject.masterlog.get_logger("zmq_monitor", "info")(msg_)

        if callback is not None:
            callback(event)

    except Exception as e:
        import traceback       
        tb_ = traceback.format_exc()
        msg_ = f"Exception in event_monitoring: {e} \n {tb_}"
        configObject.masterlog.get_logger("zmq_monitor", "error")(msg_)
        


def get_dealer_socket(identity_name, context, connect_ip, recv_timeout=10):
    # recv_timeout = seconds
    assert isinstance(identity_name, str), "identity_name should be a string."
    identity_name = identity_name.encode()
    
    socket : zmq.Socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.IDENTITY, identity_name)
    socket.setsockopt(zmq.RCVTIMEO, int(recv_timeout * 1000)) # Receive timeout
    socket.setsockopt(zmq.HEARTBEAT_IVL, int(0.5 * 1000))     # Heartbeat interval: 500 milliseconds
    socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, int(1 * 1000))   # Heartbeat timeout: 1 second
    socket.setsockopt(zmq.RECONNECT_IVL, int(1 * 1000))       # Reconnect interval: 1 second
    socket.setsockopt(zmq.RECONNECT_IVL_MAX, int(5 * 1000))   # Max reconnect interval: 5 seconds
    socket.setsockopt(zmq.HANDSHAKE_IVL, int(2 * 1000))        # Handshake interval: 2 seconds
    socket.connect(connect_ip)
    return socket, context


def set_dealer_socketopt(socket: zmq.Socket, recv_timeout=10):
    # recv_timeout = seconds
    assert socket.socket_type == zmq.DEALER, "socket should be of type ROUTER."
    socket.setsockopt(zmq.RCVHWM, 4*1000)
    socket.setsockopt(zmq.RCVBUF, 8192*1000)
    socket.setsockopt(zmq.RCVTIMEO, int(recv_timeout*1000))
    socket.setsockopt(zmq.SNDHWM, 10*1000)
    socket.setsockopt(zmq.SNDBUF, 8192*1000)
    socket.setsockopt(zmq.SNDTIMEO, 10*1000)
    socket.setsockopt(zmq.HEARTBEAT_IVL, int(0.5*1000))
    socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 2*1000)
    
    return socket


def set_pull_socketopt(socket: zmq.Socket):
    assert socket.socket_type == zmq.PULL, "socket should be of type ROUTER."
    socket.setsockopt(zmq.RCVHWM, 4*1000)
    socket.setsockopt(zmq.RCVTIMEO, 10*1000)
    socket.setsockopt(zmq.HEARTBEAT_IVL, int(0.5*1000))
    socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 2*1000)
    
    return socket

def set_router_socketopt(socket: zmq.Socket):
    assert socket.socket_type == zmq.ROUTER, "socket should be of type ROUTER."
    socket.setsockopt(zmq.RCVHWM, 4*1000)
    socket.setsockopt(zmq.RCVBUF, 8192*1000)
    socket.setsockopt(zmq.RCVTIMEO, 10*1000)
    socket.setsockopt(zmq.SNDHWM, 10*1000)
    socket.setsockopt(zmq.SNDBUF, 8192*1000)
    socket.setsockopt(zmq.SNDTIMEO, 10*1000)
    socket.setsockopt(zmq.HEARTBEAT_IVL, int(0.5*1000))
    socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 2*1000)

    socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
    socket.setsockopt(zmq.ROUTER_HANDOVER, 1) # If two clients use the same routing id when connecting to the server, the last one will take over the connection.

    return socket


def set_rep_socketopt(socket: zmq.Socket):
    assert socket.socket_type in [zmq.REP, zmq.REQ], "socket should be of type ROUTER."
    socket.setsockopt(zmq.RCVHWM, 4*1000)
    socket.setsockopt(zmq.RCVTIMEO, 10*1000)
    socket.setsockopt(zmq.SNDHWM, 10*1000)
    socket.setsockopt(zmq.SNDTIMEO, 10*1000)
    socket.setsockopt(zmq.HEARTBEAT_IVL, int(0.5*1000))
    socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 2*1000)
    
    return socket


def set_server_socket_opt(socket):
    socket.setsockopt(zmq.SNDTIMEO, 1000*10) # 10 seconds
    socket.setsockopt(zmq.HEARTBEAT_IVL, int(5*1000)) # 5 seconds
    socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, int(15*1000))
    socket.setsockopt(zmq.RCVHWM, 4*100000)
    socket.setsockopt(zmq.RCVBUF, 212992*100)
    socket.setsockopt(zmq.SNDHWM, 10*100000)
    socket.setsockopt(zmq.SNDBUF, 212992*100)
    socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
    socket.setsockopt(zmq.ROUTER_HANDOVER, 1)
    return socket

def process_event_monitor(zmq_event):

    # CONNECT CASES
    if zmq_event['event'] == zmq.EVENT_CONNECTED:
        return SocketStatus.ACTIVE
        
    elif zmq_event['event'] == zmq.EVENT_ACCEPTED:
        return SocketStatus.ACTIVE
    
    elif zmq_event['event'] == zmq.EVENT_HANDSHAKE_SUCCEEDED:
        return SocketStatus.ACTIVE

    elif zmq_event['event'] == zmq.EVENT_CONNECT_DELAYED:
        return SocketStatus.ACTIVE

    # DISCONNECT CASES
    elif zmq_event['event'] == zmq.EVENT_CLOSED:
        return SocketStatus.DISCONNECTED

    elif zmq_event['event'] == zmq.EVENT_DISCONNECTED:
        return SocketStatus.DISCONNECTED

    return None

