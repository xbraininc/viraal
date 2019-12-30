from viraal.config import pass_conf, call_if
from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)

def f(*args, **kwargs):
    return args, kwargs

class TestConfigUtils:
    def test_pass_conf(self):
        conf = OmegaConf.create({'config':{'f':{'c': 2, 'd': 10}}})

        args, kwargs = pass_conf(f, conf, 'config.f')(999,42)
        assert args[0] == 999 and args[1] == 42 and kwargs['c'] == 2 and kwargs['d'] == 10
        args, kwargs = pass_conf(f, conf, 'config.f')(999,42, c=-10)
        assert args[0] == 999 and args[1] == 42 and kwargs['c'] == -10 and kwargs['d'] == 10
    
    def test_call_if(self):
        l = [0,1]
        @call_if(True)
        def test1(a,b):
            l[0] = a
            l[1] = b
            return a,b
        r = test1(1,2)
        assert test1(1,2) == (1,2) and l[0] == 1 and l[1] == 2

        l = [0,1]
        @call_if(False)
        def test2(a,b):
            l[0] = a
            l[1] = b
            return a,b
        
        assert test2(1,2) is None and l[0] == 0 and l[1] == 1
