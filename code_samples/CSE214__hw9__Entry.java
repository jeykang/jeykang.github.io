public interface Entry<K, V> {
    public K key();
    public V value();
    public void setKey(K key);
    public void setValue(V value);
}
