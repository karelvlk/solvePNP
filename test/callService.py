
import http.client
import json

def callServiceInternal(params):
    connection = http.client.HTTPConnection('service:5000')
    headers = {'Content-type': 'application/json'}
    connection.request('POST', '/calculate', json.dumps(params), headers)
    return connection.getresponse()

def callService(params):
    response = callServiceInternal(params)
    content = response.read()
    #print(content)
    result = json.loads(content)
    return result

def callServiceCheck(input, output):
    result = callService(input)
    outputKeys = ['point', 'ypr','errors', 'points_inaccuracy', 'global_inaccuracy', 'focalLength', 'sensorSize']

    equal = True
    for key in outputKeys:
        if (key in result):
            outputValue = None
            if (key in output):
                outputValue = output[key]
            #print(result[key])
            equalKey = result[key] == outputValue
            equal = equal and equalKey
            if (not equalKey):
                print('---')
                print('key different "{key}" '.format(key=key))
                print('  result: ', result[key])
                print('  required: ', outputValue)
                print('---')

    return equal
