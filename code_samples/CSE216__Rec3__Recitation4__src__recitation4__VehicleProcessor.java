/*
The last case we must cover is that of passing an object into a method. 
In this case, we see that we are able to change the fields associated with the passed in object, 
but if we try to reassign a value to the argument itself, this reassignment is lost when the method scope is exited.
 */
package recitation4;


public class VehicleProcessor {
    
    public static void main(String args[]) {
        VehicleProcessor processor = new VehicleProcessor();
        Vehicle vehicle;
        vehicle = new Vehicle("Some name");
        System.out.println("Before calling method (vehicle = " + vehicle + ")");
        processor.process(vehicle);
        System.out.println("After calling method (vehicle = " + vehicle + ")");
        processor.processWithReferenceChange(vehicle);
        System.out.println("After calling reference-change method (vehicle = " + vehicle + ")");
    }
    
    public void process(Vehicle vehicle) {
        //System.out.println("Entered method (vehicle = " + vehicle + ")");
        vehicle.setName("A changed name");
        //System.out.println("Changed vehicle within method (vehicle = " + vehicle + ")");
        //System.out.println("Leaving method (vehicle = " + vehicle + ")");
    }
    public void processWithReferenceChange(Vehicle vehicle) {
        //System.out.println("Entered method (vehicle = " + vehicle + ")");
        vehicle = new Vehicle("A new name");
        //System.out.println("New vehicle within method (vehicle = " + vehicle + ")");
        //System.out.println("Leaving method (vehicle = " + vehicle + ")");
    }

}
