import simpy
import random
import functools
import time
import numpy as np
import pandas as pd
import logging
import os, errno
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.multioutput import MultiOutputRegressor

#try de abertura de pastas
try:
    os.makedirs('csv/delay')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
try:
    os.makedirs('csv/grant_time')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
try:
    os.makedirs("csv/pkt")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
try:
    os.makedirs("csv/overlap")
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

#Configuracao inicial
MAC_TABLE = {}
Grant_ONU_counter = {}
NUMBER_OF_OLTs = 1
NUMBER_OF_ONUs = 3
DISTANCE = 20 #Distance in kilometers
TRAFFIC = "CBR_PG"

if TRAFFIC == "poisson":
    EXPONENTS = [1160, 1450, 1740, 2030, 2320, 2610, 2900, 3190, 3480, 3770, 4060, 4350]
    CPRI_PKT = [768000]
else:
    EXPONENTS = [0]
    #CPRI_PKT = [768000, 1536000, 3072000, 3840000] #Configurações CPRI 1-4, em Kilobytes
    CPRI_PKT = [768000, 1536000, 3072000]
#load % values which represents each exponent
#loads = [25,31,37,43,50,56,62,68,75,81,87,93]
#pkt arrival distribution exponents
#exponents = [1160, 1450, 1740, 2030, 2320, 2610, 2900, 3190, 3480, 3770, 4060, 4350]
SEEDS = [20]

class ODN(object):
    """This class represents optical distribution Network."""
    def __init__(self, env, n_ONUs, n_OLTs):
        self.env = env
        self.upstream = []# upstream chanel
        self.downstream = [] # downstream chanel
        #create downstream splitter
        for i in range(n_ONUs):
            self.downstream.append(simpy.Store(env))
        for i in range(n_OLTs):
            self.upstream.append(simpy.Store(env))

    def up_latency(self, value,ONU):
        """Calculates upstream propagation delay."""
        yield self.env.timeout(ONU.delay)
        self.upstream[ONU.lamb].put(value)

    def directly_upstream(self,ONU,value):
        self.upstream[ONU.lamb].put(value)

    def down_latency(self,ONU,value):
        """Calculates downstream propagation delay."""
        yield self.env.timeout(ONU.delay)
        self.downstream[ONU.oid].put(value)

    def put_request(self, value,ONU):
        """ONU Puts the Request message in the upstream """
        self.env.process(self.up_latency(value,ONU))

    def get_request(self,lamb):
        """OLT gets the Request message from upstream  """
        return self.upstream[lamb].get()

    def put_grant(self,ONU,value):
        """OLT Puts the Grant message in the downstream """
        self.env.process(self.down_latency(ONU,value))

    def get_grant(self,ONU_id):
        """ONU gets the Grant message from downstream """
        return self.downstream[ONU_id].get()

class Packet(object):
    """ This class represents a network packet """

    def __init__(self, time, size, id, src="a", dst="z"):
        self.time = time# creation time
        self.size = size # packet size
        self.id = id # packet id
        self.src = src #packet source address
        self.dst = dst #packet destination address

    def __repr__(self):
        return "id: {}, src: {}, time: {}, size: {}".\
            format(self.id, self.src, self.time, self.size)

class PacketGenerator(object):
    """This class represents the packet generation process """
    def __init__(self, env, id, ONU, fix_pkt_size=1500, finish=float("inf")):
        self.id = id # packet id
        self.ONU = ONU
        self.env = env # Simpy Environment
        self.fix_pkt_size = fix_pkt_size # Fixed packet size
        self.finish = finish # packe end time
        self.out = None # packet generator output
        self.packets_sent = 0 # packet counter
        self.action = env.process(self.run())  # starts the run() method as a SimPy process

class CBR_PG(PacketGenerator):
    """This class represents the Constant Bit Rate packet generation process """
    def __init__(self,env, id, ONU, fix_pkt_size,interval=0.004):
        self.interval = interval
        PacketGenerator.__init__(self,env, id, ONU, fix_pkt_size)
        if fix_pkt_size == 768000:
            self.eth_overhead = 0.00001562
        elif fix_pkt_size == 1536000:
            self.eth_overhead = 0.00003004
        elif fix_pkt_size == 3072000:
            self.eth_overhead = 0.00005887
        else:
            self.eth_overhead = 0.00007329
    def run(self):
        """The generator function used in simulations.
        """
        yield self.env.timeout(random.expovariate(100))
        while self.env.now < self.finish:
            # wait for next transmission
            yield self.env.timeout(self.interval)

            npkt = self.fix_pkt_size / 1500
            npkt = int((npkt*4)/10)
            p_list = []
            for i in range(npkt):
                self.packets_sent += 1
                p = Packet(self.env.now, 1500, self.packets_sent, src=self.id)
                p_list.append(p)
                pkt_file.write("{},{},{}\n".format(self.env.now, self.interval, self.fix_pkt_size))
            self.env.timeout(self.eth_overhead)
            for p in p_list:
                self.out.put(p) # put the packet in ONU port

class poisson_PG(PacketGenerator): #Acho que está com problemas
    """This class represents the poisson distribution packet generation process """
    def __init__(self,env, id, ONU, adist, sdist, fix_pkt_size):
        self.arrivals_dist = adist #packet arrivals distribution
        self.size_dist = sdist #packet size distribution
        PacketGenerator.__init__(self,env, id, ONU, fix_pkt_size, finish=float("inf"))
    def run(self):
        """The generator function used in simulations.
        """
        while self.env.now < self.finish:
            # wait for next transmission
            arrival = self.arrivals_dist()
            yield self.env.timeout(arrival)
            self.packets_sent += 1


            if self.fix_pkt_size:
                p = Packet(self.env.now, self.fix_pkt_size, self.packets_sent, src=self.id)
                pkt_file.write("{},{},{}\n".format(self.env.now,arrival,self.fix_pkt_size))
            else:
                size = self.size_dist()
                p = Packet(self.env.now, size, self.packets_sent, src=self.id)
                pkt_file.write("{},{},{}\n".format(self.env.now,arrival,size))
            self.out.put(p) # put the packet in ONU port

class ONUPort(object):

    def __init__(self, env, ONU, qlimit=None):
        self.buffer = simpy.Store(env)#buffer
        self.grant_real_usage = simpy.Store(env) # Used in grant prediction report
        self.grant_size = 0
        self.ONU = ONU
        self.grant_final_time = 0
        self.predicted_grant = False #flag if it is a predicted grant
        self.guard_interval = 0.000001
        self.env = env
        self.out = None # ONU port output
        self.packets_rec = 0 #received pkt counter
        self.packets_drop = 0 #dropped pkt counter
        self.qlimit = qlimit #Buffer queue limit
        self.byte_size = 0  #Current size of the buffer in bytes
        self.busy = 0  # Used to track if a packet is currently being sent
        self.action = env.process(self.run())  # starts the run() method as a SimPy process
        self.pkt = None #network packet obj
        self.grant_loop = False #flag if grant time is being used
        self.current_grant_delay = []

    def get_current_grant_delay(self):
        return self.current_grant_delay
    def reset_curret_grant_delay(self):
        self.current_grant_delay = []

    def set_grant(self,grant,pred=False): #setting grant byte size and its ending
        self.grant_size = grant['grant_size']
        self.grant_final_time = grant['grant_final_time']
        self.predicted_grant = pred

    def get_pkt(self):
        """process to get the packet from the buffer   """

        try:
            pkt = (yield self.buffer.get() )#getting a packet from the buffer
            self.pkt = pkt

        except simpy.Interrupt as i:
            logging.debug("Error while getting a packet from the buffer ({})".format(i))

            pass

        if not self.grant_loop:#put the pkt back to the buffer if the grant time expired

            self.buffer.put(pkt)

    def send(self):
        """ process to send packets
        """
        self.grant_loop = True #flag if grant time is being used
        start_grant_usage = None #grant timestamp
        end_grant_usage = 0 #grant timestamp
        why_break = "ok"

        #self.current_grant_delay = []

        while self.grant_final_time > self.env.now:

            get_pkt = self.env.process(self.get_pkt())#trying to get a package in the buffer
            grant_timeout = self.env.timeout(self.grant_final_time - self.env.now)
            yield get_pkt | grant_timeout#wait for a package to be sent or the grant timeout

            if (self.grant_final_time <= self.env.now):
                #The grant time has expired
                why_break ="time expired"
                break
            if self.pkt is not None:
                pkt = self.pkt
                if not start_grant_usage:
                    start_grant_usage = self.env.now #initialized the real grant usage time
                start_pkt_usage = self.env.now ##initialized the pkt usage time

            else:
                #there is no pkt to be sent
                logging.debug("{}: there is no packet to be sent".format(self.env.now))
                why_break = "no pkt"
                break
            self.busy = 1
            self.byte_size -= pkt.size
            if self.byte_size < 0:#Prevent the buffer from being negative
                logging.debug("{}: Negative buffer".format(self.env.now))
                self.byte_size += pkt.size
                self.buffer.put(pkt)
                why_break = "negative buffer"
                break

            bits = pkt.size * 8
            sending_time = 	bits/float(10000000000) # buffer transmission time

            #To avoid fragmentation by passing the Grant window
            if env.now + sending_time > self.grant_final_time + self.guard_interval:
                self.byte_size += pkt.size

                self.buffer.put(pkt)
                why_break = "fragmentation"
                break

            #write the pkt transmission delay
            self.current_grant_delay.append(self.env.now - pkt.time)
            yield self.env.timeout(sending_time)

            delay_file.write( "{},{}\n".format( self.ONU.oid, (self.env.now - pkt.time)+self.ONU.delay ) )
            if self.predicted_grant:
                delay_prediction_file.write( "{},{}\n".format( self.ONU.oid, (self.env.now - pkt.time)+self.ONU.delay ) )

            else:
                delay_normal_file.write( "{},{}\n".format( self.ONU.oid, (self.env.now - pkt.time)+self.ONU.delay ) )


            end_pkt_usage = self.env.now
            end_grant_usage += end_pkt_usage - start_pkt_usage

            self.pkt = None

        #ending of the grant
        self.grant_loop = False #flag if grant time is being used
        if start_grant_usage and end_grant_usage > 0:# if any pkt has been sent
            #send the real grant usage
            yield self.env.timeout(self.ONU.delay) # propagation delay
            self.grant_real_usage.put( [start_grant_usage , start_grant_usage + end_grant_usage] )
        else:
            #print why_break
            #logging.debug("buffer_size:{}, grant duration:{}".format(b,grant_timeout))
            self.grant_real_usage.put([])# send a empty list



    def run(self): #run the port as a simpy process
        while True:
            yield self.env.timeout(5)


    def put(self, pkt):
        """receives a packet from the packet genarator and put it on the queue
            if the queue is not full, otherwise drop it.
        """

        self.packets_rec += 1
        tmp = self.byte_size + pkt.size
        if self.qlimit is None: #checks if the queue size is unlimited
            self.byte_size = tmp
            return self.buffer.put(pkt)
        if tmp >= self.qlimit: # chcks if the queue is full
            self.packets_drop += 1
            #return
        else:
            self.byte_size = tmp
            self.buffer.put(pkt)

class ONU(object):
    def __init__(self,distance,oid,env,lamb,channel,odn,qlimit,bucket,packet_gen,pg_param):
        self.env = env
        self.odn= odn
        self.channel = channel
        self.grant_report_store = simpy.Store(self.env) #Simpy Stores grant usage report
        self.request_container = simpy.Container(env, init=2, capacity=2)
        self.grant_report = []
        self.distance = distance #fiber distance
        self.oid = oid #ONU indentifier
        self.delay = self.distance/float(200000) # fiber propagation delay
        self.excess = 0 #difference between the size of the request and the grant
        self.newArrived = 0
        self.last_req_buffer = 0
        self.request_counter = 0
        self.pg = packet_gen(self.env, "bbmp", self, **pg_param) #creates the packet generator
        if qlimit == 0: # checks if the queue has a size limit
            queue_limit = None
        else:
            queue_limit = qlimit
        self.port = ONUPort(self.env, self, qlimit=queue_limit)#create ONU PORT
        self.pg.out = self.port #forward packet generator output to ONU port
        self.sender = self.env.process(self.ONU_sender(odn))
        self.receiver = self.env.process(self.ONU_receiver(odn))
        self.bucket = bucket #Bucket size
        self.lamb = lamb # wavelength lambda


    def ONU_receiver(self,odn):
        while True:
            # Grant stage
            grant = yield odn.get_grant(self.oid)#waiting for a grant
            pred_grant_usage_report = [] # grant prediction report list
            # real start and endtime used report to OLT
            try:
                next_grant = grant['grant_start_time'] - self.env.now #time until next grant begining
                yield self.env.timeout(next_grant)  #wait for the next grant
            except Exception as e:
                pass

            self.excess = self.last_req_buffer - grant['grant_size'] #update the excess
            self.port.set_grant(grant,False) #grant info to onu port
            if self.channel.getchannel() == 0:
                self.channel.blockchannel(self.oid)
            else:
                print ("{} - COLLISION".format(self.env.now))

            sent_pkt = self.env.process(self.port.send()) # send pkts during grant time
            yield sent_pkt # wait grant be used
            #print ("{} - arrived at OLT onu{}:".format(self.env.now + self.delay,self.oid))
            grant_usage = yield self.port.grant_real_usage.get() # get grant real utilisation
            if len(grant_usage) == 0: #debug
                logging.debug("Error in grant_usage")

            # Prediction stage
            if grant['prediction']:#check if have any predicion in the grant

                print("ONU received predictions - pred: {}".format(grant['prediction']))

                self.port.reset_curret_grant_delay()
                for pred in grant['prediction']:
                    self.channel.freechannel(self.oid)
                    # construct grant pkt
                    pred_grant = {'grant_size': grant['grant_size'], 'grant_final_time': pred[1]}
                    #wait next cycle
                    try:
                        next_grant = pred[0] - self.env.now #time until next grant begining
                        yield self.env.timeout(next_grant)  #wait for the next grant
                    except Exception as e:
                        logging.debug("{}: pred {}, gf {}".format(self.env.now,pred,grant['grant_final_time']))
                        logging.debug("Error while waiting for the next grant ({})".format(e))
                        break

                    self.port.set_grant(pred_grant,True) #grant info to onu port
                    if self.channel.getchannel() == 0:
                        self.channel.blockchannel(self.oid)
                    else:
                        print ("{} - COLLISION in Pred".format(self.env.now))
                    sent_pkt = self.env.process(self.port.send())#sending predicted messages
                    yield sent_pkt # wait grant be used
                    grant_usage = yield self.port.grant_real_usage.get() # get grant real utilisation
                    if len(grant_usage) > 0: # filling grant prediction report list
                        pred_grant_usage_report.append(grant_usage)
                        #logging.debug("{}:pred={},usage={}".format(self.env.now,pred,grant_usage))
                    else:
                        logging.debug("{}:Error in pred_grant_usage".format(self.env.now))
                        break
            # grant mean squared errors
            if len(pred_grant_usage_report) > 0:
                delay = self.port.get_current_grant_delay()
                if len(delay) == 0:
                    logging.debug("{}:Error in current grant delay".format(self.env.now))
                    # print pred_grant_usage_report
                    # print delay
                    delay.append(-1)
                len_usage = len(pred_grant_usage_report)
                mse_start = mse(np.array(pred_grant_usage_report)[:,0],np.array(grant['prediction'][:len_usage])[:,0])
                mse_end = mse(np.array(pred_grant_usage_report)[:,1],np.array(grant['prediction'][:len_usage])[:,1])
                mse_file.write("{},{},{}\n".format(mse_start,mse_end,np.mean(delay)))

            self.port.reset_curret_grant_delay()
            self.channel.freechannel(self.oid)

            #Signals the end of grant processing to allow new requests
            yield self.grant_report_store.put(pred_grant_usage_report)

    def ONU_sender(self, odn):
        """A process which checks the queue size and send a REQUEST message to OLT"""
        while True:
            # send a REQUEST only if the queue size is greater than the bucket size
            #yield self.request_container.get(1)
            if self.port.byte_size >= self.bucket:
                requested_buffer = self.port.byte_size #gets the size of the buffer that will be requested
                #update the size of the current/last buffer REQUEST
                self.last_req_buffer = requested_buffer
                # creating request message
                msg = {'text':"ONU %s sent this REQUEST for %.6f at %f" %
                    (self.oid,self.port.byte_size, self.env.now),'buffer_size':requested_buffer,'ONU':self}
                odn.put_request((msg),self)# put the request message in the odn

                # Wait for the grant processing to send the next request
                self.grant_report = yield self.grant_report_store.get()
                #yield self.env.timeout(2*self.delay)
            else: # periodic check delay
                #yield self.request_container.put(1)
                yield self.env.timeout(self.delay)

class DBA(object):
    """DBA Parent class, heritated by every kind of DBA"""
    def __init__(self,env,max_grant_size,grant_store):
        self.env = env
        self.max_grant_size = max_grant_size
        self.grant_store = grant_store
        self.guard_interval = 0.000001

class IPACT(DBA):
    def __init__(self,env,max_grant_size,grant_store):
        DBA.__init__(self,env,max_grant_size,grant_store)
        self.counter = simpy.Resource(self.env, capacity=1)#create a queue of requests to DBA


    def dba(self,ONU,buffer_size):
        with self.counter.request() as my_turn:
            """ DBA only process one request at a time """
            yield my_turn
            time_stamp = self.env.now # timestamp dba starts processing the request
            delay = ONU.delay # oneway delay

            # check if max grant size is enabled
            if self.max_grant_size > 0 and buffer_size > self.max_grant_size:
                buffer_size = self.max_grant_size
            bits = buffer_size * 8
            sending_time = 	bits/float(10000000000) #buffer transmission time
            grant_time = delay + sending_time
            grant_final_time = self.env.now + grant_time # timestamp for grant end
            counter = Grant_ONU_counter[ONU.oid] # Grant message counter per ONU
            # write grant log
            grant_time_file.write( "{},{},{},{},{},{},{},{}\n".format(MAC_TABLE['olt'], MAC_TABLE[ONU.oid],"02", time_stamp,counter, ONU.oid,self.env.now,grant_final_time) )
            # construct grant message
            grant = {'ONU':ONU,'grant_size': buffer_size, 'grant_final_time': grant_final_time, 'prediction': None}
            self.grant_store.put(grant) # send grant to OLT
            Grant_ONU_counter[ONU.oid] += 1

            # timeout until the end of grant to then get next grant request
            yield self.env.timeout(delay+grant_time + self.guard_interval)

class PD_DBA(DBA):
    def __init__(self,env,max_grant_size,grant_store,window=20,predict=5,model="ols"):
        DBA.__init__(self,env,max_grant_size,grant_store)
        self.counter = simpy.Resource(self.env, capacity=1)#create a queue of requests to DBA
        self.window = window    # past observations window size
        self.predict = predict # number of predictions
        self.grant_history = [] #grant history per ONU (training set)
        self.predictions_array = []
        for i in range(NUMBER_OF_ONUs):
            # training unit
            self.grant_history.insert(i, {'counter': [], 'start': [], 'end': []})
        #Implementing the model
        if model == "ols":
            reg = linear_model.LinearRegression()
        else:
            reg = linear_model.Ridge(alpha=.5)

        self.model = MultiOutputRegressor(reg)

    def predictions_schedule(self,predictions):
        predictions = map(list,predictions)
        predictions_cp = list(predictions)
        if len(self.predictions_array) > 0:
            self.predictions_array = list(filter(lambda x: x[0] > self.env.now, self.predictions_array))

        predictions_array_cp = list(self.predictions_array)
        predictions_array_cp +=  predictions
        predictions_array_cp.sort()

        over = False
        j = 1
        for interval1 in predictions_array_cp[:-1]:
            for interval2 in predictions_array_cp[j:]:
                if interval1[1] > interval2[0]:
                    overlap_file.write("{}\n".format(interval1[1] - interval2[0]))
                    over = True
                    if interval1 in predictions:
                        index = predictions.index(interval1)
                        new_interval = [ interval1[0] , interval2[0]]
                        predictions_cp[ index ] = new_interval

                    elif interval2 in predictions:
                        index = predictions.index(interval2)
                        new_interval = [ interval1[1], interval2[1] ]
                        predictions_cp[ index ] = new_interval

                else:
                    break
                j+=1

        if over:
            predictions = None
        else:
            predictions = predictions_cp
            self.predictions_array += predictions
        return predictions
    
    def drop_overlap(self,predictions,ONU):
        predcp = list(predictions)
        j = 1
        #drop: if there are overlaps between the predictions
        for p in predcp[:-1]:
            for q in predcp[j:]:
                if p[1] + ONU.delay  > q[0]:
                    predictions = None
                    break

            j+=1
        #drop: if there is overlap between standard grant and first prediction
        if predictions is not None and (self.grant_history[ONU.oid]['end'][-1] +ONU.delay+ self.guard_interval) > predictions[0][0]:
            predictions = None

        return predictions
    
    def predictor(self, ONU):
        # check if there's enough observations to fill window

        if len( self.grant_history[ONU.oid]['start'] ) >= self.window :
            #reduce the grant history to the window size
            self.grant_history[ONU.oid]['start'] = self.grant_history[ONU.oid]['start'][-self.window:]
            self.grant_history[ONU.oid]['end'] = self.grant_history[ONU.oid]['end'][-self.window:]
            self.grant_history[ONU.oid]['counter'] = self.grant_history[ONU.oid]['counter'][-self.window:]
            df_tmp = pd.DataFrame(self.grant_history[ONU.oid]) # temp dataframe w/ past grants
            
            print(df_tmp.to_string())

            # create a list of the next p Grants that will be predicted
            X_pred = np.arange(
                self.grant_history[ONU.oid]['counter'][-1] +1,
                self.grant_history[ONU.oid]['counter'][-1] + 1 + self.predict
            ).reshape(-1,1)


            # model fitting
            self.model.fit( np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp[['start','end']] )
            pred = self.model.predict(X_pred) # predicting start and end

            predictions = list(pred)
            predcp = list(predictions)
            #print(predcp)

            j = 1
            #drop: if there are overlaps between the predictions
            bucket_time = (ONU.bucket*8)/float(10000000000)
            #print(ONU.delay+bucket_time)

            for p in predcp[:-1]:
                for q in predcp[j:]:
                    if p[1] + NUMBER_OF_ONUs*(ONU.delay+bucket_time)  > q[0]:
                        #print("a")
                        predictions = None
                        break
                j+=1
                

            #drop: if there is overlap between standard grant and first prediction
            if predictions is not None and (self.grant_history[ONU.oid]['end'][-1] +ONU.delay+ self.guard_interval) > predictions[0][0]:
                predictions = None

            #drop if there is overlap with predictions array
            if predictions is not None:

                if len(self.predictions_array) == 0:
                    self.predictions_array += predictions
                else:
                    self.predictions_array = list(filter(lambda x: x[0] > self.env.now, self.predictions_array))
                    predcp = list(predictions)
                    newpred = []
                    drop = False
                    for interval1 in predcp:
                        for interval2 in self.predictions_array:
                            if interval1[1] > interval2[0]:
                                if interval1[0] < interval2[0]:
                                    drop = True
                                    break


                            if interval1[0] < interval2[1]:
                                if interval1[1] > interval2[1]:
                                    drop = True
                                    break
                        if drop == False:
                            newpred.append(interval1)
                        else:
                            break
                    if len(newpred)> 0:
                        predictions = newpred
                        self.predictions_array += predictions
                        self.predictions_array = sorted(self.predictions_array,key=lambda x: x[0])
                    else:
                        predictions = None

            print("PD_DBA class predictor - pred: {}".format(predictions))

            return predictions

        else:
            return  None
        
    def dba(self,ONU,buffer_size):
        with self.counter.request() as my_turn:
            """ DBA only process one request at a time """
            yield my_turn
            time_stamp = self.env.now
            delay = ONU.delay

            if len(ONU.grant_report) > 0:
                # if predictions where utilized, update history with real grant usage
                for report in ONU.grant_report:
                    self.grant_history[ONU.oid]['start'].append(report[0])
                    self.grant_history[ONU.oid]['end'].append(report[1])
                    self.grant_history[ONU.oid]['counter'].append( self.grant_history[ONU.oid]['counter'][-1] + 1  )

            # check if max grant size is enabled
            if self.max_grant_size > 0 and buffer_size > self.max_grant_size:
                buffer_size = self.max_grant_size
            bits = buffer_size * 8
            sending_time = 	bits/float(10000000000) #buffer transmission time10g
            grant_time = delay + sending_time # one way delay + transmission time
            grant_final_time = self.env.now + grant_time # timestamp for grant end
            counter = Grant_ONU_counter[ONU.oid] # Grant message counter per ONU
            
            # Update grant history with grant requested
            if len(self.predictions_array) > 0:
                print("Time now: {}".format(self.env.now))
                print("Array predictions: {}".format(self.predictions_array))
                self.predictions_array = list(filter(lambda x: x[0] > self.env.now, self.predictions_array))
                
                print(self.predictions_array)

                if len(self.predictions_array) > 0:
                    if (grant_final_time+ONU.delay+self.guard_interval) > self.predictions_array[0][0]:
                        bits = ONU.bucket * 8
                        sending_time = 	bits/float(10000000000) #buffer transmission time10g
                        grant_time = delay + sending_time # one way delay + transmission time
                        grant_final_time = self.env.now + grant_time # timestamp for grant end

            self.grant_history[ONU.oid]['start'].append(self.env.now)
            self.grant_history[ONU.oid]['end'].append(grant_final_time)
            if len(self.grant_history[ONU.oid]['counter']) > 0:
                self.grant_history[ONU.oid]['counter'].append( self.grant_history[ONU.oid]['counter'][-1] + 1  )
            else:
                self.grant_history[ONU.oid]['counter'].append( 1 )

            #PREDICTIONS
            predictions = self.predictor(ONU) # start predictor process

            #drop if the predictions have overlap
            # if predictions is not None:
            #     predictions = self.drop_overlap(predictions,ONU)


            grant_time_file.write( "{},{},{},{},{},{},{},{}\n".format(MAC_TABLE['olt'], MAC_TABLE[ONU.oid],"02", time_stamp, counter, ONU.oid,self.env.now,grant_final_time))
            # construct grant message
            grant = {'ONU':ONU,'grant_size': buffer_size, 'grant_final_time': grant_final_time, 'prediction': predictions}

            self.grant_store.put(grant) # send grant to OLT

            # timeout until the end of grant to then get next grant request
            yield self.env.timeout(grant_time + delay + self.guard_interval)

class MPD_DBA(DBA):
    def __init__(self,env,max_grant_size,grant_store,window=20,predict=5,model="ols"):
        DBA.__init__(self,env,max_grant_size,grant_store)
        self.counter = simpy.Resource(self.env, capacity=1)#create a queue of requests to DBA
        self.window = window    # past observations window size
        self.predict = predict # number of predictions
        self.next_grant = 0
        self.grant_history = [] #grant history per ONU (training set)
        self.predictions_array = []
        self.predictions_counter_array = []
        for i in range(NUMBER_OF_ONUs):
            # training unit
            self.grant_history.insert(i, {'counter': [], 'start': [], 'end': []})
            self.predictions_counter_array.append(0)
        #Implementing the model
        if model == "ols":
            reg = linear_model.LinearRegression()
        else:
            reg = linear_model.Ridge(alpha=.5)

        self.model = MultiOutputRegressor(reg)

    def drop_overlap(self,predictions,ONU):
        predcp = list(predictions)
        j = 1
        #drop: if there are overlaps between the predictions
        for p in predcp[:-1]:
            for q in predcp[j:]:
                if p[1] + ONU.delay  > q[0]:
                    predictions = None
                    break

            j+=1
        #drop: if there is overlap between standard grant and first prediction
        if predictions is not None and (self.grant_history[ONU.oid]['end'][-1] +ONU.delay+ self.guard_interval) > predictions[0][0]:
            predictions = None


        return predictions

    def predictor(self, ONU_id):
        # check if there's enough observations to fill window

        if len( self.grant_history[ONU_id]['start'] ) >= self.window :
            #reduce the grant history to the window size
            self.grant_history[ONU_id]['start'] = self.grant_history[ONU_id]['start'][-self.window:]
            self.grant_history[ONU_id]['end'] = self.grant_history[ONU_id]['end'][-self.window:]
            self.grant_history[ONU_id]['counter'] = self.grant_history[ONU_id]['counter'][-self.window:]
            df_tmp = pd.DataFrame(self.grant_history[ONU_id]) # temp dataframe w/ past grants
            # create a list of the next p Grants that will be predicted
            X_pred = np.arange(self.grant_history[ONU_id]['counter'][-1] +1, self.grant_history[ONU_id]['counter'][-1] + 1 + self.predict).reshape(-1,1)

            # model fitting
            self.model.fit( np.array( df_tmp['counter'] ).reshape(-1,1) , df_tmp[['start','end']] )
            pred = self.model.predict(X_pred) # predicting start and end

            predictions = list(pred)
            #predictions = self.predictions_schedule(predictions)

            return predictions

        else:
            return  None


    def dba(self,ONU,buffer_size):
        with self.counter.request() as my_turn:
            """ DBA only process one request at a time """
            yield my_turn
            time_stamp = self.env.now
            delay = ONU.delay

            if len(ONU.grant_report) > 0:
                # if predictions where utilized, update history with real grant usage
                for report in ONU.grant_report:
                    self.grant_history[ONU.oid]['start'].append(report[0])
                    self.grant_history[ONU.oid]['end'].append(report[1])
                    self.grant_history[ONU.oid]['counter'].append( self.grant_history[ONU.oid]['counter'][-1] + 1  )

            # check if max grant size is enabled
            if self.max_grant_size > 0 and buffer_size > self.max_grant_size:
                buffer_size = self.max_grant_size
            bits = buffer_size * 8
            sending_time = 	bits/float(10000000000) #buffer transmission time10g
            grant_time = delay + sending_time # one way delay + transmission time
            ini = max(self.env.now,self.next_grant)
            grant_final_time = ini + grant_time # timestamp for grant end

            # Update grant history with grant requested
            self.grant_history[ONU.oid]['start'].append(self.env.now)
            self.grant_history[ONU.oid]['end'].append(grant_final_time)
            if len(self.grant_history[ONU.oid]['counter']) > 0:
                self.grant_history[ONU.oid]['counter'].append( self.grant_history[ONU.oid]['counter'][-1] + 1  )
            else:
                self.grant_history[ONU.oid]['counter'].append( 1 )

            if self.predictions_counter_array[ONU.oid] > 0:
                self.predictions_counter_array[ONU.oid] -= 1
            else:
                #PREDICTIONS
                predictions = self.predictor(ONU.oid) # start predictor process

                #drop if the predictions have overlap
                if predictions is not None:
                    predictions = self.drop_overlap(predictions,ONU)

                if predictions is not None:
                    self.predictions_counter_array[ONU.oid] = len(predictions)


                #grant_time_file.write( "{},{},{}\n".format(ONU.oid,self.env.now,grant_final_time))
                # construct grant message
                grant = {'ONU': ONU, 'grant_size': buffer_size, 'grant_final_time': grant_final_time, 'prediction': predictions}

                self.grant_store.put(grant) # send grant to OLT

                # timeout until the end of grant to then get next grant request
                self.next_grant = grant_final_time + delay + self.guard_interval

class OLT(object):
    """Optical line terminal"""
    def __init__(self,env,lamb,odn,max_grant_size,dba,window,predict,model,numberONUs):
        self.env = env
        self.lamb = lamb
        self.grant_store = simpy.Store(self.env) # grant communication between processes

        #choosing algorithms
        if dba == 'pd_dba':
            self.dba = PD_DBA(self.env, max_grant_size, self.grant_store,window,predict,model)
        elif dba == 'mpd_dba':
            self.dba = MPD_DBA(self.env, max_grant_size, self.grant_store,window,predict,model)
        else:
            self.dba = IPACT(self.env, max_grant_size, self.grant_store)

        self.receiver = self.env.process(self.OLT_receiver(odn)) # process for receiving requests
        self.sender = self.env.process(self.OLT_sender(odn)) # process for sending grant

    def OLT_sender(self,odn):
        """A process which sends a grant message to ONU"""
        while True:
            grant = yield self.grant_store.get() # receive grant from dba
            odn.put_grant(grant['ONU'],grant) # send grant to odn

    def OLT_receiver(self,odn):
        """A process which receives a request message from the ONUs."""
        while True:
            request = yield odn.get_request(self.lamb) #get a request message
            #print("Received Request from ONU {} at {}".format(request['ONU'].oid, self.env.now))
            # send request to DBA
            self.env.process(self.dba.dba(request['ONU'],request['buffer_size']))

class collisionDetection(object):
    def __init__(self):
        self.chanel=0
        self.whoIsUsing = None
    def getchannel(self):
        return self.chanel
    def blockchannel(self,oid):
        self.chanel = 1
        self.whoIsUsing = oid
    def freechannel(self,oid):
        if self.whoIsUsing == oid:
            self.chanel = 0

for seed in SEEDS:
    for exp in EXPONENTS:
        for pkt_size in CPRI_PKT:
            FILENAME = "PD_DBA-dist{}-{}ONUs-{}OLTs-{}-exp{}-pkt{}".format(DISTANCE,NUMBER_OF_ONUs, NUMBER_OF_OLTs, TRAFFIC, exp, pkt_size)
            if "PD_DBA" in FILENAME:
                FILENAME = FILENAME+"-w20-p8"
            #abertura de arquivos
            delay_file = open("csv/delay/{}-s{}-delay.csv".format(FILENAME, seed),"w")
            delay_prediction_file = open("csv/delay/{}-s{}-delay_pred.csv".format(FILENAME, seed),"w")
            delay_normal_file = open("csv/delay/{}-s{}-delay_normal.csv".format(FILENAME, seed),"w")
            grant_time_file = open("csv/grant_time/{}-s{}-grant_time.csv".format(FILENAME, seed),"w")
            pkt_file = open("csv/pkt/{}-s{}-pkt.csv".format(FILENAME, seed),"w")
            overlap_file = open("csv/overlap/{}-s{}-overlap.csv".format(FILENAME, seed),"w")
            mse_file = open("csv/{}-s{}-mse.csv".format(FILENAME, seed), "w")

            delay_file.write("ONU_id,delay\n")
            delay_normal_file.write("ONU_id,delay\n")
            delay_prediction_file.write("ONU_id,delay\n")
            grant_time_file.write("source address,destination address,opcode,timestamp,counter,ONU_id,start,end\n")
            pkt_file.write("timestamp,adist,size\n")
            overlap_file.write("interval\n")
            mse_file.write("mse_start,mse_end,delay\n")

            #inicio de execução
            random.seed(seed)
            env = simpy.Environment()
            odn = ODN(env, NUMBER_OF_ONUs, NUMBER_OF_OLTs)
            
            #Parametros de trafego
            if TRAFFIC == "poisson":
                packet_generator = poisson_PG
                pg_params = {"adist":functools.partial(random.expovariate, exp), "sdist":None, "fix_pkt_size":pkt_size}
            else:
                packet_generator = CBR_PG
                pg_params = {"fix_pkt_size":pkt_size} 

            #ONU creation
            ONU_list = []
            lamb = 0
            channel = collisionDetection()

            for i in range(NUMBER_OF_ONUs):
                MAC_TABLE[i] = "00:00:00:00:{}:{}".format(random.randint(0x00, 0xff),random.randint(0x00, 0xff))
                Grant_ONU_counter[i] = 0

            for i in range(NUMBER_OF_ONUs):
                ONU_list.append(
                    ONU(DISTANCE, i, env, lamb, channel, odn, 0, 27000, packet_generator, pg_params)
                )

            #OLT creation
            olt = OLT(env, lamb, odn, 0, 'pd_dba', 20, 8, 'ols', NUMBER_OF_ONUs)
            MAC_TABLE['olt'] = "ff:ff:ff:ff:00:01"
            logging.info("Starting Simulator")
            env.run(until=30) #Tempo de duracao simulado, em Segundos

            #Closing files
            delay_file.close()
            delay_normal_file.close()
            delay_prediction_file.close()
            grant_time_file.close()
            pkt_file.close()
            overlap_file.close()
            mse_file.close()